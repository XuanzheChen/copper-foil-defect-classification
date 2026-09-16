#!/usr/bin/env python3
"""Measure single-image model inference latency on images below one directory.

This is a temporary benchmark utility. It intentionally focuses on forward-pass
latency rather than classification accuracy.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.convnextv2 import convnextv2_tiny


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_CHECKPOINT = "pretrained-weights/convnextv2_tiny_1k_224_ema.pt"


@dataclass(frozen=True)
class ImageRecord:
    path: Path


class DirectImageDataset(Dataset):
    def __init__(self, root: Path, transform: transforms.Compose) -> None:
        self.transform = transform
        self.records = [ImageRecord(path) for path in self._image_paths(root)]
        if not self.records:
            raise ValueError(f"No supported images found below {root}")

    @staticmethod
    def _image_paths(directory: Path) -> list[Path]:
        return sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        record = self.records[index]
        with Image.open(record.path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(record.path)


def parse_args() -> argparse.Namespace:
    default_checkpoint = os.environ.get(
        "PRETRAINED_WEIGHT", str(PROJECT_ROOT / DEFAULT_CHECKPOINT)
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(default_checkpoint))
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--warmup-images",
        type=int,
        default=20,
        help="Number of single-image forward passes used only for warmup.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of timed images; 0 means all images.",
    )
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


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot calculate percentile of an empty list")
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def forward_once(
    model: torch.nn.Module,
    image: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> None:
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            model(image)


def run_latency_benchmark(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    warmup_images: int,
    max_images: int,
) -> list[float]:
    # Warmup also uses batch_size=1 so kernel selection/cache behavior matches
    # the actual timed path.
    warmup_iterator = iter(loader)
    for _ in range(warmup_images):
        try:
            images, _ = next(warmup_iterator)
        except StopIteration:
            break
        images = images.to(device, non_blocking=False)
        forward_once(model, images, device, amp)

    sync_if_needed(device)

    latencies_ms: list[float] = []
    for images, _ in loader:
        if max_images > 0 and len(latencies_ms) >= max_images:
            break

        # Loading, PIL decode, preprocessing and H2D transfer happen outside
        # the timer. The measured value is model forward latency only.
        images = images.to(device, non_blocking=False)
        sync_if_needed(device)
        start = time.perf_counter()

        forward_once(model, images, device, amp)

        sync_if_needed(device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return latencies_ms


def main() -> None:
    args = parse_args()
    if args.input_size < 32:
        raise ValueError("--input-size must be at least 32")
    if args.warmup_images < 0:
        raise ValueError("--warmup-images must be non-negative")
    if args.max_images < 0:
        raise ValueError("--max-images must be non-negative")
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
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    model = load_model(args.checkpoint, device)

    latencies_ms = run_latency_benchmark(
        model=model,
        loader=loader,
        device=device,
        amp=args.amp,
        warmup_images=args.warmup_images,
        max_images=args.max_images,
    )
    if not latencies_ms:
        raise RuntimeError("No images were timed")

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    p50_ms = percentile(latencies_ms, 0.50)
    p95_ms = percentile(latencies_ms, 0.95)
    p99_ms = percentile(latencies_ms, 0.99)
    sequential_fps = 1000.0 / mean_ms

    print(f"timed_images: {len(latencies_ms)}")
    print("batch_size: 1")
    print("num_workers: 0")
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"amp: {args.amp and device.type == 'cuda'}")
    print("timing_scope: model forward only (excludes image decode/preprocess/H2D)")
    print(f"latency_mean_ms: {mean_ms:.3f}")
    print(f"latency_p50_ms: {p50_ms:.3f}")
    print(f"latency_p95_ms: {p95_ms:.3f}")
    print(f"latency_p99_ms: {p99_ms:.3f}")
    print(f"sequential_fps: {sequential_fps:.2f}")


if __name__ == "__main__":
    main()
