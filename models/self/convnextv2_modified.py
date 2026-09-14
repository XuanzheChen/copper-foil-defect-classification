import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from ..utils import LayerNorm, GRN
import sys

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)

        return x * y.expand_as(x)
    
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        avg_out = self.fc2(self.relu(self.fc1(self.global_avg_pool(x).view(b, c))))
        max_out = self.fc2(self.relu(self.fc1(self.global_max_pool(x).view(b, c))))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(2, 2, kernel_size=kernel_size, padding=kernel_size//2, groups=2, bias=False),
        #     nn.Conv2d(2, 1, kernel_size=1, bias=False)
        # )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv(torch.cat([avg_out, max_out], dim=1))
        return x * self.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class GRN_SE(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        
        self.fc1 = nn.Linear(dim, dim // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, dim, bias=False)

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        b, _, _, c = Nx.shape
        Nx = self.relu(self.fc2(self.relu(self.fc1(Nx.view(b, c))))).view(b, 1, 1, c)
        return self.gamma * (x * Nx) + self.beta + x
    
class GRN_CBAM(nn.Module):
    def __init__(self, dim, reduction=16, kernel_size=7):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

        self.fc1 = nn.Linear(dim, dim // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim // reduction, dim, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)

    def ca(self, x):
        b, _, _, c = x.shape
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        Nx = self.fc2(self.relu(self.fc1(Nx.view(b, c))))
        return x * self.relu(Nx).view(b, 1, 1, c)

    def sa(self, x):
        SA_avg_out = torch.mean(x, dim=3, keepdim=True)
        SA_max_out, _ = torch.max(x, dim=3, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([SA_avg_out, SA_max_out], dim=3).permute(0, 3, 1, 2))).permute(0, 2, 3, 1)

    def forward(self, x):
        input = x
        x = self.ca(x)
        x = self.sa(x)
        return self.gamma * x + self.beta + input
    
# class GRN_SA(nn.Module): # 通道空间注意力并行
#     def __init__(self, dim):
#         super().__init__()
#         self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
#         self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

#     def forward(self, x):
#         input = x
#         Gx_c = torch.norm(x, p=2, dim=(1,2), keepdim=True)
#         Nx_c = Gx_c / (Gx_c.mean(dim=-1, keepdim=True) + 1e-6)
#         Gx_s = torch.norm(x, p=2, dim=-1, keepdim=True)
#         Nx_s = Gx_s / (Gx_s.mean(dim=(1,2), keepdim=True) + 1e-6)
#         x = x * Nx_c * Nx_s
#         return self.gamma * x + self.beta + input 
    
class GRN_SA(nn.Module): # 先通道再空间
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.conv = nn.Conv2d(1, 1, kernel_size=7, padding=3)

    def forward(self, x):
        input = x
        Gx_c = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx_c = Gx_c / (Gx_c.mean(dim=-1, keepdim=True) + 1e-6)
        x = x * Nx_c
        Gx_s = torch.norm(x, p=2, dim=-1, keepdim=True)
        Nx_s = Gx_s / (Gx_s.mean(dim=(1,2), keepdim=True) + 1e-6)
        Nx_s = self.conv(Nx_s.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = x * Nx_s
        return self.gamma * x + self.beta + input


class Block(nn.Module):
    """ ConvNeXtV2 Block with optional attention modules.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0., grn_module=None, block_module=None, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = self._init_GRN(grn_module, 4 * dim)
        
        self.module = self._init_attention_module(block_module, 4 * dim, kernel_size=kernel_size)

        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _init_attention_module(self, module, dims, kernel_size=7):
        module_dict = {
            'SE': SEBlock(channels=dims),
            'CBAM': CBAMBlock(channels=dims),
            'SA': SpatialAttention(kernel_size=kernel_size),
        }
        
        if module is None:
            return None
        elif module in module_dict:
            return module_dict[module]
        else:
            raise ValueError(f"Invalid module type: '{module}'. Expected one of {list(module_dict.keys())} or None.")
        
    def _init_GRN(self, module, dims):
        module_dict = {
            'GRN_SE': GRN_SE(dim=dims),
            'GRN_CBAM': GRN_CBAM(dim=dims),
            'GRN_SA': GRN_SA(dim=dims),
        }
        
        if module is None:
            return GRN(dims)
        elif module in module_dict:
            return module_dict[module]
        else:
            raise ValueError(f"Invalid module type: '{module}'. Expected one of {list(module_dict.keys())} or None.")

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)

        if self.module is not None:
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
            x = self.module(x)
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return input + self.drop_path(x)

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1., 
                 module_GRN=None, module_Block=None
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        grn_module=module_GRN, block_module=module_Block) 
                for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
class ConvNeXtV2_FPN(nn.Module):
    """ ConvNeXt V2 with FPN for multi-scale feature fusion
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        fpn_channels (int): Number of channels in FPN features. Default: 256
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1., 
                 module_GRN=None, module_Block=None,
                 fpn_channels=256, block_k_size=7):
        super().__init__()
        self.depths = depths
        self.dims = dims
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        grn_module=module_GRN, block_module=module_Block, kernel_size=block_k_size) 
                for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, fpn_channels, kernel_size=1),
                nn.GELU()
            ) for dim in dims
        ])
        
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=7, padding=3, groups=fpn_channels),
                nn.Conv2d(fpn_channels, fpn_channels, kernel_size=1),
                nn.GroupNorm(1, fpn_channels, eps=1e-5),
                nn.GELU()
            ) for _ in range(len(dims)-1)
        ])

        self.neck_weights = init_weights(len(dims))
        self.neck_norms = nn.ModuleList([
            nn.LayerNorm(fpn_channels, eps=1e-6) for dim in dims
        ])

        self.norm = nn.LayerNorm(fpn_channels, eps=1e-6)
        self.head = nn.Linear(fpn_channels, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        
        laterals = []
        for conv, feat in zip(self.lateral_convs, features):
            x = conv(feat)
            laterals.append(x)
        
        fpn_features = [laterals[-1]]
        for i in range(len(laterals)-2, -1, -1):
            upsampled = F.interpolate(
                fpn_features[0], 
                size=laterals[i].shape[2:],
                mode='bilinear',
                align_corners=False
            )
            fused = laterals[i] + upsampled
            fused = self.fpn_convs[i](fused)
            fpn_features.insert(0, fused)
        
        normalized_weights = LiNormWeight(self.neck_weights)
        pooled_features = sum(w * n(f.mean([-2, -1]))
                           for w, f, n in zip(normalized_weights, fpn_features, self.neck_norms))

        return self.norm(pooled_features)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class ConvNeXtV2_Fusion(nn.Module):
    """ ConvNeXt V2 with Multi-scale Fusion for Classification """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1., 
                 module_GRN=None, module_Block=None,
                 block_k_size=7):
        super().__init__()
        self.depths = depths
        self.dims = dims
        
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                  grn_module=module_GRN, block_module=module_Block, 
                  kernel_size=block_k_size) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.neck_gamma = nn.Parameter(torch.zeros(1, dims[0] + dims[-1]))
        self.neck_beta = nn.Parameter(torch.zeros(1, dims[0] + dims[-1]))

        self.norm = nn.LayerNorm(dims[0] + dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[0] + dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        
        # fusing_feature = torch.cat([ # 平均池化
        #     torch.mean(features[0], dim=(2, 3)),
        #     torch.mean(features[-1], dim=(2, 3))
        # ], dim=1)
        # fusing_feature = torch.cat([ # L2池化
        #     torch.norm(features[0], p=2, dim=(2, 3), keepdim=True).flatten(1),
        #     torch.norm(features[-1], p=2, dim=(2, 3), keepdim=True).flatten(1)
        # ],dim=1)
        fusing_feature = torch.cat([ # 最大池化
            torch.max(features[0], dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0].flatten(1),
            torch.max(features[-1], dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0].flatten(1)
        ], dim=1)
        Nx = fusing_feature / (fusing_feature.mean(dim=1, keepdim=True) + 1e-6)
        fused = self.neck_gamma * (Nx * fusing_feature) + self.neck_beta + fusing_feature
        
        return self.norm(fused)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def init_weights(num):
    base = torch.linspace(0.1, 1.0, steps=num)
    noise = torch.randn(num) * 0.01
    return nn.Parameter(base + noise)

def LiNormWeight(weights, dim=0, eps=1e-6):
    return weights / (torch.sum(weights, dim=dim, keepdim=True) + eps)

class ConvNeXtV2_BiFPN(nn.Module):
    """ConvNeXt V2 with BiFPN (minimal modification version)"""
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1., 
                 module_GRN=None, module_Block=None,
                 fpn_channels=256):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                      grn_module=module_GRN, block_module=module_Block) 
                for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, fpn_channels, kernel_size=1),
                nn.GELU()
            ) for dim in dims
        ])
        
        self.bifpn = BiFPN_Layer(fpn_channels)
        
        self.neck_weights = nn.Parameter(torch.ones(len(dims)))
        self.neck_norms = nn.ModuleList([
            nn.LayerNorm(fpn_channels, eps=1e-6) for _ in dims
        ])
        self.norm = nn.LayerNorm(fpn_channels, eps=1e-6)
        self.head = nn.Linear(fpn_channels, num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        fpn_features = self.bifpn(laterals)
        
        normalized_weights = LiNormWeight(self.neck_weights)
        pooled_features = sum(
            w * n(f.mean([-2, -1])) 
            for w, f, n in zip(normalized_weights, fpn_features, self.neck_norms)
        )
        return self.norm(pooled_features)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class BiFPN_Layer(nn.Module):
    """BiFPN实现（仅核心双向融合）"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.top_down_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(3)
        ])
        
        self.bottom_up_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2)) for _ in range(3)
        ])
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 7, padding=3, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(1, channels),
            nn.GELU()
        )

    def forward(self, features):
        p2, p3, p4, p5 = features

        w = LiNormWeight(self.top_down_weights[0])
        p4 = self.fusion_conv(w[0] * p4 + w[1] * F.interpolate(p5, size=p4.shape[2:]))
        w = LiNormWeight(self.top_down_weights[1], dim=0)
        p3 = self.fusion_conv(w[0] * p3 + w[1] * F.interpolate(p4, size=p3.shape[2:]))
        w = LiNormWeight(self.top_down_weights[2], dim=0)
        p2 = self.fusion_conv(w[0] * p2 + w[1] * F.interpolate(p3, size=p2.shape[2:]))
        w = LiNormWeight(self.bottom_up_weights[0], dim=0)
        p3 = self.fusion_conv(w[0] * p3 + w[1] * F.avg_pool2d(p2, kernel_size=2))
        w = LiNormWeight(self.bottom_up_weights[1], dim=0)
        p4 = self.fusion_conv(w[0] * p4 + w[1] * F.avg_pool2d(p3, kernel_size=2))
        w = LiNormWeight(self.bottom_up_weights[2], dim=0)
        p5 = self.fusion_conv(w[0] * p5 + w[1] * F.avg_pool2d(p4, kernel_size=2))
        
        return [p2, p3, p4, p5]

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_se_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], module_Block='SE', **kwargs)
    return model

def convnextv2_cbam_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], module_Block='CBAM', **kwargs)
    return model

def convnextv2_grn_se_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], module_GRN='GRN_SE', **kwargs)
    return model

def convnextv2_grn_cbam_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], module_GRN='GRN_CBAM', **kwargs)
    return model

def convnextv2_grn_sa_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], module_GRN='GRN_SA', **kwargs)
    return model

def convnextv2_fpn_tiny(**kwargs):
    model = ConvNeXtV2_FPN(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_fusion_tiny(**kwargs):
    model = ConvNeXtV2_Fusion(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_fpn_tiny_k3(**kwargs):
    model = ConvNeXtV2_FPN(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], block_k_size=3, **kwargs)
    return model

def convnextv2_fpn_grn_se_tiny(**kwargs):
    model = ConvNeXtV2_FPN(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], module_GRN='GRN_SE', **kwargs)
    return model

def convnextv2_fpn_grn_cbam_tiny(**kwargs):
    model = ConvNeXtV2_FPN(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], module_GRN='GRN_CBAM', **kwargs)
    return model

def convnextv2_fpn_grn_sa_tiny(**kwargs):
    model = ConvNeXtV2_FPN(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], module_GRN='GRN_SA', **kwargs)
    return model

def convnextv2_fusion_grn_sa_tiny(**kwargs):
    model = ConvNeXtV2_Fusion(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], module_GRN='GRN_SA', **kwargs)
    return model

def convnextv2_bifpn_tiny(**kwargs):
    model = ConvNeXtV2_BiFPN(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def convnextv2_bifpn_grn_sa_tiny(**kwargs):
    model = ConvNeXtV2_BiFPN(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], module_GRN='GRN_SA', **kwargs)
    return model

def convnextv2_sa_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], module_Block='SA', **kwargs)
    return model

def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def convnextv2_large(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model