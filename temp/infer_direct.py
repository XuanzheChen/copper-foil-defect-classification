#!/usr/bin/env python3
"""Benchmark batch_size=1 inference and generate a local infer_speed.md report."""

from __future__ import annotations

import argparse
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.convnextv2 import convnextv2_tiny


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_CHECKPOINT = "pretrained-weights/convnextv2_tiny_1k_224_ema.pt"
DEFAULT_REPORT = PROJECT_ROOT / "infer_speed.md"


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
    parser.add_argument("--warmup-images", type=int, default=50)
    parser.add_argument(
        "--timed-images",
        "--max-images",
        dest="timed_images",
        type=int,
        default=1000,
        help="Timed requests per benchmark; 0 means each source image once.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def collect_image_paths(root: Path) -> list[Path]:
    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported images found below {root}")
    return paths


def build_transform(input_size: int) -> transforms.Compose:
    resize_size = round(input_size / 0.875)
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model = convnextv2_tiny(num_classes=1000)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    w = pos - lo
    return ordered[lo] * (1 - w) + ordered[hi] * w


def summarize(values: list[float]) -> dict[str, float]:
    mean_ms = sum(values) / len(values)
    return {
        "fps": 1000.0 / mean_ms,
        "mean": mean_ms,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def forward(model, tensor, device, amp):
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            return model(tensor)


def prepare_tensor(path: Path, transform, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0).to(device, non_blocking=False)


def benchmark_model_only(model, tensor, device, amp, warmup, count) -> list[float]:
    for _ in range(warmup):
        forward(model, tensor, device, amp)
    sync(device)

    times = []
    for _ in range(count):
        sync(device)
        start = time.perf_counter()
        forward(model, tensor, device, amp)
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def end_to_end_request(path: Path, transform, model, device, amp) -> int:
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB"))
    tensor = tensor.unsqueeze(0).to(device, non_blocking=False)
    logits = forward(model, tensor, device, amp)
    return int(logits.argmax(dim=1).item())


def benchmark_end_to_end(paths, transform, model, device, amp, warmup, count):
    for i in range(warmup):
        end_to_end_request(paths[i % len(paths)], transform, model, device, amp)
    sync(device)

    times = []
    for i in range(count):
        sync(device)
        start = time.perf_counter()
        end_to_end_request(paths[i % len(paths)], transform, model, device, amp)
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def run_command(command: list[str]) -> str:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False,
            timeout=15, env=env,
        )
        return result.stdout.strip() or result.stderr.strip() or "Unavailable"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "Unavailable"


def command_snapshots(data_path: Path) -> dict[str, str]:
    return {
        "nvidia-smi": run_command(["nvidia-smi"]),
        "lscpu": run_command(["lscpu"]),
        "free -h": run_command(["free", "-h"]),
        "lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINT": run_command(
            ["lsblk", "-o", "NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINT"]
        ),
        f"df -hT {data_path.resolve()}": run_command(
            ["df", "-hT", str(data_path.resolve())]
        ),
        "nvidia-smi topo -m": run_command(["nvidia-smi", "topo", "-m"]),
    }


def cpu_model(lscpu_text: str) -> str:
    for line in lscpu_text.splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "Unavailable"


def report_text(args, device, paths, count, model_stats, e2e_stats, snapshots) -> str:
    gpu_name = "N/A"
    vram = "N/A"
    capability = "N/A"
    sms = "N/A"
    if device.type == "cuda" and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(device)
        gpu_name = p.name
        vram = f"{p.total_memory / 1024**3:.2f} GiB"
        capability = f"{p.major}.{p.minor}"
        sms = str(p.multi_processor_count)

    nvidia_text = snapshots["nvidia-smi"]
    driver = re.search(r"Driver Version:\s*([0-9.]+)", nvidia_text)
    driver_cuda = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_text)
    driver = driver.group(1) if driver else "Unavailable"
    driver_cuda = driver_cuda.group(1) if driver_cuda else "Unavailable"

    def row(name, s):
        return (
            f"| {name} | {s['fps']:.2f} | {s['mean']:.3f} | {s['p50']:.3f} | "
            f"{s['p95']:.3f} | {s['p99']:.3f} | {s['min']:.3f} | {s['max']:.3f} |"
        )

    lines = [
        "# Inference Speed Benchmark",
        "",
        f"Generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "",
        "This file was generated locally by `temp/infer_direct.py`. The script does not upload or commit this report to Git.",
        "",
        "## Results",
        "",
        "Both measurements use `batch_size=1`, sequential requests, and exclude model/checkpoint loading time.",
        "",
        "| Scope | FPS | Mean ms | P50 ms | P95 ms | P99 ms | Min ms | Max ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        row("Model forward only", model_stats),
        row("End-to-end", e2e_stats),
        "",
        "- **Model forward only:** preprocessed `[1, 3, H, W]` tensor already resides on the inference device; only `model(tensor)` is timed.",
        "- **End-to-end:** file open/decode → RGB → resize/crop/normalize → CPU→device transfer → model forward → argmax → scalar result on CPU.",
        "- Repeated source files may benefit from the operating-system page cache when timed requests exceed the number of unique images.",
        "",
        "## Benchmark Configuration",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Model | ConvNeXt V2 Tiny |",
        f"| Checkpoint | `{args.checkpoint}` |",
        f"| Input size | {args.input_size} × {args.input_size} |",
        "| Batch size | 1 |",
        "| Request mode | Sequential |",
        f"| Device | `{device}` |",
        f"| AMP / FP16 autocast | {args.amp and device.type == 'cuda'} |",
        f"| Unique source images | {len(paths)} |",
        f"| Timed requests per benchmark | {count} |",
        f"| Warmup requests per benchmark | {args.warmup_images} |",
        f"| Source directory | `{args.data_path.resolve()}` |",
        "",
        "## Hardware and Software",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| GPU | {gpu_name} |",
        f"| GPU VRAM | {vram} |",
        f"| GPU SM count | {sms} |",
        f"| GPU compute capability | {capability} |",
        f"| NVIDIA driver | {driver} |",
        f"| Driver-supported CUDA | {driver_cuda} |",
        f"| CPU | {cpu_model(snapshots['lscpu'])} |",
        f"| Logical CPUs | {os.cpu_count()} |",
        f"| OS | {platform.platform()} |",
        f"| Python | {platform.python_version()} |",
        f"| PyTorch | {torch.__version__} |",
        f"| TorchVision | {torchvision.__version__} |",
        f"| PyTorch CUDA runtime | {torch.version.cuda or 'N/A'} |",
        f"| cuDNN | {torch.backends.cudnn.version() or 'N/A'} |",
        "",
        "## Raw System Information",
        "",
    ]

    for command, output in snapshots.items():
        lines += [
            f"### `{command}`",
            "",
            "```text",
            output,
            "```",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    if args.input_size < 32:
        raise ValueError("--input-size must be at least 32")
    if args.warmup_images < 0 or args.timed_images < 0:
        raise ValueError("warmup/timed image counts must be non-negative")
    if not args.data_path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {args.data_path}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    paths = collect_image_paths(args.data_path)
    random.Random(args.seed).shuffle(paths)
    count = len(paths) if args.timed_images == 0 else args.timed_images
    transform = build_transform(args.input_size)
    model = load_model(args.checkpoint, device)
    tensor = prepare_tensor(paths[0], transform, device)

    print("Running model-only benchmark...")
    model_stats = summarize(benchmark_model_only(
        model, tensor, device, args.amp, args.warmup_images, count
    ))

    print("Running end-to-end benchmark...")
    e2e_stats = summarize(benchmark_end_to_end(
        paths, transform, model, device, args.amp, args.warmup_images, count
    ))

    snapshots = command_snapshots(args.data_path)
    report = report_text(
        args, device, paths, count, model_stats, e2e_stats, snapshots
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    print(f"model_forward_fps: {model_stats['fps']:.2f}")
    print(f"end_to_end_fps: {e2e_stats['fps']:.2f}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
