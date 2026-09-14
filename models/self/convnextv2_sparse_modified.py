# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

from ..utils import (
    LayerNorm,
    MinkowskiLayerNorm,
    MinkowskiGRN,
    MinkowskiDropPath
)
from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
    MinkowskiGELU,
    SparseTensor,
)
from MinkowskiOps import (
    to_sparse,
)

from .convnextv2_modified import GRN_SE, GRN_SA, GRN_CBAM

class Block(nn.Module):
    """ Sparse ConvNeXtV2 Block. 

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., D=3, grn_module=None):
        super().__init__()
        self.dwconv = MinkowskiDepthwiseConvolution(dim, kernel_size=7, bias=True, dimension=D)
        self.norm = MinkowskiLayerNorm(dim, 1e-6)
        self.pwconv1 = MinkowskiLinear(dim, 4 * dim)   
        self.act = MinkowskiGELU()
        self.grn = self._init_GRN(grn_module, 4  * dim)
        self.pwconv2 = MinkowskiLinear(4 * dim, dim)
        self.drop_path = MinkowskiDropPath(drop_path)
        self.grn_module = grn_module

    def _init_GRN(self, module, dims):
        module_dict = {
            'GRN_SE': GRN_SE(dim=dims),
            'GRN_CBAM': GRN_CBAM(dim=dims),
            'GRN_SA': GRN_SA(dim=dims),
        }
        
        if module is None:
            return MinkowskiGRN(dims)
        elif module in module_dict:
            return module_dict[module]
        else:
            raise ValueError(f"Invalid module type: '{module}'. Expected one of {list(module_dict.keys())} or None.")

    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.grn_module is None:
            x = self.grn(x)
        else:
            x = x.dense()[0]
            x = self.grn(x.permute(0,2,3,1))
            print(x.shape)
            x = to_sparse(x.permute(0,3,1,2))
            x = SparseTensor(
                x.F,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x

class SparseConvNeXtV2(nn.Module):
    """ Sparse ConvNeXtV2.
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, 
                 in_chans=3, 
                 num_classes=1000, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768], 
                 drop_path_rate=0., 
                 D=3,
                 module_GRN=None):
        super().__init__()
        self.depths = depths
        self.num_classes = num_classes
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                MinkowskiLayerNorm(dims[i], eps=1e-6),
                MinkowskiConvolution(dims[i], dims[i+1], kernel_size=2, stride=2, bias=True, dimension=D)
            )
            self.downsample_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], D=D, grn_module=module_GRN) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, MinkowskiConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiDepthwiseConvolution):
            trunc_normal_(m.kernel, std=.02)
            nn.init.constant_(m.bias, 0)
        if isinstance(m, MinkowskiLinear):
            trunc_normal_(m.linear.weight, std=.02)
            nn.init.constant_(m.linear.bias, 0)

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)

    def forward(self, x, mask):
        num_stages = len(self.stages)
        mask = self.upsample_mask(mask, 2**(num_stages-1))        
        mask = mask.unsqueeze(1).type_as(x)
        
        # patch embedding
        x = self.downsample_layers[0](x)
        x *= (1.-mask)
        
        # sparse encoding
        x = to_sparse(x)
        for i in range(4):
            x = self.downsample_layers[i](x) if i > 0 else x
            x = self.stages[i](x)
        
        # densify
        x = x.dense()[0]
        return x
    