#!/usr/bin/env python3
"""Measure steady-state end-to-end single-image inference latency.

This temporary benchmark intentionally simulates a sequential deployment path:
open/decode one image -> RGB conversion -> preprocessing -> CPU-to-device copy ->
model forward -> argmax -> prediction returned to CPU.

Model/checkpoint loading is intentionally excluded because a deployed service
normally loads the model once and then handles many image requests.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.convnextv2 import convnextv2_tiny


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_CHECKPOINT = "pretrained-weights/convnextv2_tiny_1k_224_ema.pt"


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
        default=50,
        help="Number of untimed end-to-end requests used for warmup.",
    )
    parser.add_argument(
        "--timed-images",
        "--max-images",
        dest="timed_images",
        type=int,
        default=1000,
        help=(
            "Number of timed single-image requests. Images are reused cyclically "
            "when the directory contains fewer images. Set 0 to process each "
            "source image exactly once. Default: 1000."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to shuffle source image order before benchmarking.",
    )
    return parser.parse_args()


def collect_image_paths(root: Path) -> list[Path]:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found below {root}")
    return paths


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


def run_single_request(
    image_path: Path,
    transform: transforms.Compose,
    model: torch.nn.Module,
    device: torch.device,
    amp: bool,
) -> int:
    # Image.open + convert covers file access and image decoding. The transform
    # covers resize/crop/tensor conversion/normalization.
    with Image.open(image_path) as image:
        image_tensor = transform(image.convert("RGB"))

    # A real single-image classifier normally adds the batch dimension just
    # before inference, then transfers the tensor to the inference device.
    image_tensor = image_tensor.unsqueeze(0).to(device, non_blocking=False)

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(image_tensor)

    # Include minimal deployment post-processing and returning the result to CPU.
    prediction = logits.argmax(dim=1)
    return int(prediction.item())


def request_path(paths: list[Path], request_index: int) -> Path:
    return paths[request_index % len(paths)]


def run_end_to_end_benchmark(
    paths: list[Path],
    transform: transforms.Compose,
    model: torch.nn.Module,
    device: torch.device,
    amp: bool,
    warmup_images: int,
    timed_images: int,
) -> list[float]:
    # Warm up the same end-to-end path used during the actual benchmark. This
    # excludes cold-start effects from model/kernel initialization and better
    # reflects a continuously running deployed classifier.
    for index in range(warmup_images):
        run_single_request(
            request_path(paths, index), transform, model, device, amp
        )
    sync_if_needed(device)

    request_count = len(paths) if timed_images == 0 else timed_images
    latencies_ms: list[float] = []

    for index in range(request_count):
        # Ensure the previous request has fully completed before measuring the
        # next one. This models sequential batch_size=1 request handling.
        sync_if_needed(device)
        start = time.perf_counter()

        run_single_request(
            request_path(paths, index), transform, model, device, amp
        )

        # prediction.item() already requires the result on CPU, but an explicit
        # synchronize keeps the timing boundary unambiguous for CUDA workloads.
        sync_if_needed(device)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    return latencies_ms


def main() -> None:
    args = parse_args()
    if args.input_size < 32:
        raise ValueError("--input-size must be at least 32")
    if args.warmup_images < 0:
        raise ValueError("--warmup-images must be non-negative")
    if args.timed_images < 0:
        raise ValueError("--timed-images must be non-negative")
    if not args.data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_path}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    image_paths = collect_image_paths(args.data_path)
    random.Random(args.seed).shuffle(image_paths)

    transform = build_transform(args.input_size)
    model = load_model(args.checkpoint, device)

    latencies_ms = run_end_to_end_benchmark(
        paths=image_paths,
        transform=transform,
        model=model,
        device=device,
        amp=args.amp,
        warmup_images=args.warmup_images,
        timed_images=args.timed_images,
    )
    if not latencies_ms:
        raise RuntimeError("No images were timed")

    total_seconds = sum(latencies_ms) / 1000.0
    mean_ms = sum(latencies_ms) / len(latencies_ms)
    p50_ms = percentile(latencies_ms, 0.50)
    p95_ms = percentile(latencies_ms, 0.95)
    p99_ms = percentile(latencies_ms, 0.99)
    min_ms = min(latencies_ms)
    max_ms = max(latencies_ms)
    end_to_end_fps = len(latencies_ms) / total_seconds

    print(f"source_images: {len(image_paths)}")
    print(f"timed_requests: {len(latencies_ms)}")
    print(f"warmup_requests: {args.warmup_images}")
    print("batch_size: 1")
    print("request_mode: sequential")
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"input_size: {args.input_size}")
    print(f"amp: {args.amp and device.type == 'cuda'}")
    print(
        "timing_scope: file open/decode -> RGB -> resize/crop/normalize -> "
        "H2D -> model forward -> argmax -> CPU result"
    )
    print(f"end_to_end_total_seconds: {total_seconds:.3f}")
    print(f"end_to_end_latency_mean_ms: {mean_ms:.3f}")
    print(f"end_to_end_latency_p50_ms: {p50_ms:.3f}")
    print(f"end_to_end_latency_p95_ms: {p95_ms:.3f}")
    print(f"end_to_end_latency_p99_ms: {p99_ms:.3f}")
    print(f"end_to_end_latency_min_ms: {min_ms:.3f}")
    print(f"end_to_end_latency_max_ms: {max_ms:.3f}")
    print(f"end_to_end_fps: {end_to_end_fps:.2f}")


if __name__ == "__main__":
    main()
