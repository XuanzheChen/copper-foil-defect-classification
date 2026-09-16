#!/usr/bin/env python3
"""Run deployment-style inference on all images below one directory."""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.convnextv2 import convnextv2_tiny


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train", "val", "test", "eval", "evaluation"}
DEFAULT_CHECKPOINT = "pretrained-weights/convnextv2_tiny_1k_224_ema.pt"


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int | None


class DirectImageDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.root = root
        self.transform = transform
        self.class_names, self.records = self._collect_records()

        if not self.records:
            raise ValueError(f"No supported images found below {root}")

    def _collect_records(self) -> tuple[list[str], list[ImageRecord]]:
        top_level_directories = sorted(
            path for path in self.root.iterdir() if path.is_dir()
        )
        split_directories = [
            path
            for path in top_level_directories
            if path.name.lower() in SPLIT_NAMES
        ]

        if split_directories:
            class_names = sorted(
                {
                    class_directory.name
                    for split_directory in split_directories
                    for class_directory in split_directory.iterdir()
                    if class_directory.is_dir()
                }
            )
            class_to_index = {
                class_name: index for index, class_name in enumerate(class_names)
            }
            records = []
            for split_directory in split_directories:
                for class_name, label in class_to_index.items():
                    class_directory = split_directory / class_name
                    if class_directory.is_dir():
                        records.extend(
                            ImageRecord(path, label)
                            for path in self._image_paths(class_directory)
                        )
            return class_names, records

        if top_level_directories:
            class_names = [path.name for path in top_level_directories]
            class_to_index = {
                class_name: index for index, class_name in enumerate(class_names)
            }
            records = []
            for class_directory in top_level_directories:
                records.extend(
                    ImageRecord(path, class_to_index[class_directory.name])
                    for path in self._image_paths(class_directory)
                )
            return class_names, records

        return [], [ImageRecord(path, None) for path in self._image_paths(self.root)]

    @staticmethod
    def _image_paths(directory: Path) -> list[Path]:
        return sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        label = -1 if record.label is None else record.label
        return tensor, label, str(record.path)


def parse_args() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    default_checkpoint = os.environ.get(
        "PRETRAINED_WEIGHT", str(script_directory / DEFAULT_CHECKPOINT)
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(default_checkpoint))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=5)
    return parser.parse_args()


def build_transform(input_size: int) -> transforms.Compose:
    resize_size = round(input_size / 0.875)
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model = convnextv2_tiny(num_classes=1000)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    warmup_batches: int,
) -> tuple[int, int, float, int]:
    iterator = iter(loader)
    for _ in range(warmup_batches):
        try:
            images, _, _ = next(iterator)
        except StopIteration:
            break
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                model(images.to(device, non_blocking=device.type == "cuda"))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    total_images = 0
    labeled_images = 0
    correct_images = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                logits = model(images)
        predictions = logits.argmax(dim=1).cpu()
        valid_labels = labels >= 0
        labeled_images += int(valid_labels.sum())
        if valid_labels.any():
            correct_images += int((predictions[valid_labels] == labels[valid_labels]).sum())
        total_images += images.shape[0]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    return correct_images, labeled_images, elapsed, total_images


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.input_size < 32:
        raise ValueError("--input-size must be at least 32")
    if not args.data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_path}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    dataset = DirectImageDataset(args.data_path, build_transform(args.input_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model = load_model(args.checkpoint, device)
    correct, labeled, elapsed, total = run_inference(
        model, loader, device, args.amp, args.warmup_batches
    )

    print(f"images: {total}")
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"class folders: {len(dataset.class_names)}")
    if dataset.class_names:
        print(f"classes: {', '.join(dataset.class_names)}")
    if labeled:
        print(f"accuracy: {correct / labeled * 100:.2f}% ({correct}/{labeled})")
        if len(dataset.class_names) != 1000:
            print(
                "accuracy note: folder indices are compared with ImageNet-1K "
                "output indices; provide a matching label map for semantic accuracy"
            )
    else:
        print("accuracy: N/A (images are not arranged in labeled class folders)")
    print(f"fps: {total / elapsed:.2f}")
    print(f"elapsed_seconds: {elapsed:.3f}")


if __name__ == "__main__":
    main()
