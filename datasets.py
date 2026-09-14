# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from PIL import Image

import cv2
import numpy as np

class HSVEnhance:
    def __init__(self, h_gain=0.05, s_gain=0.1, v_gain=0.1):
        self.h_gain = h_gain
        self.s_gain = s_gain
        self.v_gain = v_gain

    def __call__(self, img):
        img = np.array(img)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

        h = img_hsv[..., 0]
        s = img_hsv[..., 1]
        v = img_hsv[..., 2]

        h = (h + np.random.uniform(-self.h_gain * 360, self.h_gain * 360)) % 360
        s = np.clip(s * (1 + np.random.uniform(-self.s_gain, self.s_gain)), 0, 255)
        v = np.clip(v * (1 + np.random.uniform(-self.v_gain, self.v_gain)), 0, 255)

        img_hsv[..., 0] = h
        img_hsv[..., 1] = s
        img_hsv[..., 2] = v

        img_hsv = img_hsv.astype(np.uint8)
        img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        return Image.fromarray(img_rgb)

class GaussianNoise:
    def __init__(self, mean=0, sigma=20):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, image):        
        if isinstance(image, np.ndarray):
            img = image
        else:
            img = np.array(image)

        row, col, ch = img.shape if len(img.shape) == 3 else (*img.shape, 1)
        gauss = np.random.normal(self.mean, self.sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        gauss = gauss * np.random.rand()
        noisy_image = img + gauss
        noisy_image = np.clip(noisy_image, 0, 255)
        noisy_image = noisy_image.astype(np.uint8)
        if not isinstance(image, np.ndarray):
            noisy_image = Image.fromarray(noisy_image)

        return noisy_image

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = len([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # 训练集：随机裁剪一个比例1:1的区域
        transformlist = []
        if args.use_grayscale:
            transformlist.append(transforms.Grayscale(num_output_channels=3))
        
        transformlist.append(
            transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0), ratio=(1.0, 1.0))
        )
        if args.HSVEnhance:
            h_gain, s_gain, v_gain = args.HSVE_factors
            transformlist.append(
                HSVEnhance(h_gain=h_gain, s_gain=s_gain, v_gain=v_gain)
            )
        if args.add_Gauss_Noise:
            GN_mean, GN_std = args.GN_factors
            transformlist.append(
                GaussianNoise(mean=GN_mean, sigma=GN_std)
            )

        # 添加其他增强
        transformlist.append(
            create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
                mean=mean,
                std=std,
            )
        )
        
        transform = transforms.Compose(transformlist)
        return transform

    else:
        # 验证集：中心裁剪并统一尺寸
        t = []
        if resize_im:
            if args.input_size >= 384:  
                t.append(
                    transforms.Resize(
                        (args.input_size, args.input_size), interpolation=transforms.InterpolationMode.BICUBIC
                    )
                )
                print(f"Warping {args.input_size} size input images...")
            else:
                if args.crop_pct is None:
                    args.crop_pct = 224 / 256
                size = int(args.input_size / args.crop_pct)
                t.append(
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)
                )
                t.append(transforms.CenterCrop(args.input_size))  # 中心裁剪

        if args.use_grayscale:
            t.insert(0, transforms.Grayscale(num_output_channels=3))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))

        return transforms.Compose(t)
