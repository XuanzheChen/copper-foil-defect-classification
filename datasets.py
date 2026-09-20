# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Dataset and preprocessing utilities for the copper-foil baseline.

The formal copper-foil pipeline intentionally keeps the native 200 x 200
pixel scale. Images are admitted only when their original size is exactly
200 x 200, decoded to real 8-bit grayscale before augmentation, and expanded
back to three identical channels only immediately before tensor conversion.
"""

import csv
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


FORMAL_IMAGE_SIZE = (200, 200)
TRANSLATION_PADDING = 6
BRIGHTNESS_RANGE = (0.95, 1.05)
CONTRAST_RANGE = (0.92, 1.08)
GAUSSIAN_NOISE_SIGMA_RANGE = (0.0, 0.005)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_grayscale_image(path):
    """Load an admitted image and explicitly decode it as true grayscale.

    In particular, palette PNGs must not reach ToTensor() in PIL mode "P".
    The size assertion is repeated here defensively in case a file changes
    after the dataset was scanned.
    """
    with open(path, "rb") as file_obj:
        with Image.open(file_obj) as img:
            if img.size != FORMAL_IMAGE_SIZE:
                raise ValueError(
                    f"Formal samples must be {FORMAL_IMAGE_SIZE}, got "
                    f"{img.size} for {path!r}."
                )
            return img.convert("L")


class RandomBrightness:
    """Apply grayscale brightness with a factor sampled from a fixed range."""

    def __init__(self, factor_range=BRIGHTNESS_RANGE):
        self.factor_range = tuple(factor_range)

    def __call__(self, img):
        if img.mode != "L":
            raise ValueError(
                f"Brightness must run in grayscale mode 'L', got {img.mode!r}."
            )
        factor = torch.empty(1).uniform_(*self.factor_range).item()
        return TF.adjust_brightness(img, factor)

    def __repr__(self):
        return f"{self.__class__.__name__}(factor_range={self.factor_range})"


class RandomContrast:
    """Apply grayscale contrast with a factor sampled from a fixed range."""

    def __init__(self, factor_range=CONTRAST_RANGE):
        self.factor_range = tuple(factor_range)

    def __call__(self, img):
        if img.mode != "L":
            raise ValueError(
                f"Contrast must run in grayscale mode 'L', got {img.mode!r}."
            )
        factor = torch.empty(1).uniform_(*self.factor_range).item()
        return TF.adjust_contrast(img, factor)

    def __repr__(self):
        return f"{self.__class__.__name__}(factor_range={self.factor_range})"


class GrayscaleToRGB:
    """Replicate one grayscale channel into three identical RGB channels."""

    def __call__(self, img):
        if img.mode != "L":
            raise ValueError(
                f"RGB expansion expects grayscale mode 'L', got {img.mode!r}."
            )
        return img.convert("RGB")


class SharedChannelGaussianNoise:
    """Add one Gaussian noise map to all three channels before normalization."""

    def __init__(self, sigma_range=GAUSSIAN_NOISE_SIGMA_RANGE):
        self.sigma_range = tuple(sigma_range)

    def __call__(self, x):
        if not torch.is_tensor(x) or x.ndim != 3 or x.shape[0] != 3:
            raise ValueError(
                "SharedChannelGaussianNoise expects Tensor[3, H, W]."
            )
        if not (torch.equal(x[0], x[1]) and torch.equal(x[1], x[2])):
            raise ValueError(
                "Expected identical RGB grayscale channels before Gaussian noise."
            )
        if torch.any(x < 0.0) or torch.any(x > 1.0):
            raise ValueError("Gaussian noise expects an input tensor in [0, 1].")

        sigma = torch.empty((), dtype=x.dtype, device=x.device).uniform_(
            *self.sigma_range
        )
        noise = torch.randn_like(x[:1]) * sigma
        x = (x + noise.expand_as(x)).clamp(0.0, 1.0)

        if not (torch.equal(x[0], x[1]) and torch.equal(x[1], x[2])):
            raise RuntimeError("Shared Gaussian noise broke channel equality.")
        if not torch.isfinite(x).all():
            raise RuntimeError("Gaussian noise produced NaN or Inf values.")
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(sigma_range={self.sigma_range})"


class CopperFoilImageFolder(datasets.ImageFolder):
    """ImageFolder that excludes non-200x200 samples before formal loading."""

    def __init__(self, root, transform, exclusion_report_path):
        super().__init__(
            root=root,
            transform=transform,
            loader=load_grayscale_image,
        )
        self.exclusion_report_path = Path(exclusion_report_path)

        kept_samples = []
        excluded = []

        for path, target in self.samples:
            with Image.open(path) as img:
                width, height = img.size

            if (width, height) != FORMAL_IMAGE_SIZE:
                excluded.append(
                    {
                        "path": path,
                        "class": self.classes[target],
                        "width": width,
                        "height": height,
                        "reason": "non_200x200",
                    }
                )
            else:
                kept_samples.append((path, target))

        self.samples = kept_samples
        self.imgs = kept_samples
        self.targets = [target for _, target in kept_samples]
        self.excluded_non_200x200 = excluded
        self._write_exclusion_report()

    def _write_exclusion_report(self):
        self.exclusion_report_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["path", "class", "width", "height", "reason"]
        with self.exclusion_report_path.open(
            "w", newline="", encoding="utf-8"
        ) as report:
            writer = csv.DictWriter(report, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.excluded_non_200x200)


def _validate_formal_input_size(args):
    input_size = getattr(args, "input_size", FORMAL_IMAGE_SIZE[0])
    if input_size != FORMAL_IMAGE_SIZE[0]:
        raise ValueError(
            "Formal copper-foil preprocessing requires --input_size 200. "
            "Resize is intentionally forbidden."
        )


def _build_exclusion_report_path(args, is_train):
    base = Path(getattr(args, "output_dir", "") or os.getcwd())
    split = "train" if is_train else "eval"
    return (
        base
        / "preprocessing_reports"
        / split
        / "excluded_non_200x200.csv"
    )


def build_dataset(is_train, args):
    """Build the formal copper-foil folder dataset.

    The formal preprocessing applies to folder-based training/evaluation data.
    The legacy CIFAR option is retained only for upstream compatibility and is
    not part of the copper-foil baseline.
    """
    if args.data_set == "CIFAR":
        dataset = datasets.CIFAR100(
            args.data_path,
            train=is_train,
            transform=transforms.ToTensor(),
            download=True,
        )
        return dataset, 100

    if args.data_set not in {"IMNET", "image_folder"}:
        raise NotImplementedError(f"Unsupported dataset type: {args.data_set}")

    _validate_formal_input_size(args)
    transform = build_transform(is_train, args)

    if args.data_set == "IMNET":
        root = os.path.join(args.data_path, "train" if is_train else "val")
    else:
        root = args.data_path if is_train else args.eval_data_path

    if not root:
        raise ValueError("A dataset root is required for formal preprocessing.")

    report_path = _build_exclusion_report_path(args, is_train)
    dataset = CopperFoilImageFolder(
        root=root,
        transform=transform,
        exclusion_report_path=report_path,
    )

    if args.data_set == "image_folder":
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        nb_classes = len(dataset.classes)

    print("Transform =")
    for transform_step in transform.transforms:
        print(transform_step)
    print("---------------------------")
    print(f"Formal dataset root: {root}")
    print(f"Admitted samples: {len(dataset)}")
    print(f"Excluded non-200x200 samples: {len(dataset.excluded_non_200x200)}")
    print(f"Exclusion report: {dataset.exclusion_report_path}")
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    """Build the fixed formal train or evaluation preprocessing pipeline."""
    _validate_formal_input_size(args)

    if is_train:
        return transforms.Compose(
            [
                transforms.Pad(
                    TRANSLATION_PADDING,
                    padding_mode="reflect",
                ),
                transforms.RandomCrop(FORMAL_IMAGE_SIZE),
                RandomBrightness(BRIGHTNESS_RANGE),
                RandomContrast(CONTRAST_RANGE),
                GrayscaleToRGB(),
                transforms.ToTensor(),
                SharedChannelGaussianNoise(GAUSSIAN_NOISE_SIGMA_RANGE),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            GrayscaleToRGB(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
