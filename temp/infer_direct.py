#!/usr/bin/env python3
"""Benchmark batch_size=1 inference on CPU/GPU and generate infer_speed.md."""

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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class BenchmarkMode:
    key: str
    label: str
    device: torch.device
    amp: bool


def parse_args() -> argparse.Namespace:
    default_checkpoint = os.environ.get(
        "PRETRAINED_WEIGHT", str(PROJECT_ROOT / DEFAULT_CHECKPOINT)
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(default_checkpoint))
    parser.add_argument("--input-size", type=int, default=224)
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
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("cpu", "cuda", "cuda-amp"),
        default=None,
        help=(
            "Benchmark modes. Default: CPU FP32 plus GPU FP32 and GPU AMP/FP16 "
            "when CUDA is available. Example: --modes cpu cuda"
        ),
    )
    return parser.parse_args()


def resolve_modes(requested: list[str] | None) -> list[BenchmarkMode]:
    cuda_available = torch.cuda.is_available()
    keys = requested or (["cpu", "cuda", "cuda-amp"] if cuda_available else ["cpu"])

    if any(key.startswith("cuda") for key in keys) and not cuda_available:
        raise RuntimeError("CUDA benchmark mode was requested but CUDA is not available")

    modes: list[BenchmarkMode] = []
    for key in keys:
        if key == "cpu":
            modes.append(BenchmarkMode("cpu", "CPU FP32", torch.device("cpu"), False))
        elif key == "cuda":
            modes.append(BenchmarkMode("cuda", "GPU FP32", torch.device("cuda"), False))
        elif key == "cuda-amp":
            modes.append(
                BenchmarkMode(
                    "cuda-amp",
                    "GPU AMP (FP16 autocast)",
                    torch.device("cuda"),
                    True,
                )
            )
    return modes


def collect_image_paths(root: Path) -> list[Path]:
    paths = sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
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


def forward(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            return model(tensor)


def prepare_tensor(
    path: Path,
    transform: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0).to(device, non_blocking=False)


def benchmark_model_only(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    amp: bool,
    warmup: int,
    count: int,
) -> list[float]:
    for _ in range(warmup):
        forward(model, tensor, device, amp)
    sync(device)

    times: list[float] = []
    for _ in range(count):
        sync(device)
        start = time.perf_counter()
        forward(model, tensor, device, amp)
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def end_to_end_request(
    path: Path,
    transform: transforms.Compose,
    model: torch.nn.Module,
    device: torch.device,
    amp: bool,
) -> int:
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB"))

    tensor = tensor.unsqueeze(0).to(device, non_blocking=False)
    logits = forward(model, tensor, device, amp)
    return int(logits.argmax(dim=1).item())


def benchmark_end_to_end(
    paths: list[Path],
    transform: transforms.Compose,
    model: torch.nn.Module,
    device: torch.device,
    amp: bool,
    warmup: int,
    count: int,
) -> list[float]:
    for i in range(warmup):
        end_to_end_request(paths[i % len(paths)], transform, model, device, amp)
    sync(device)

    times: list[float] = []
    for i in range(count):
        sync(device)
        start = time.perf_counter()
        end_to_end_request(paths[i % len(paths)], transform, model, device, amp)
        sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def benchmark_mode(
    mode: BenchmarkMode,
    args: argparse.Namespace,
    paths: list[Path],
    transform: transforms.Compose,
    count: int,
) -> dict[str, dict[str, float]]:
    if mode.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    model = load_model(args.checkpoint, mode.device)
    tensor = prepare_tensor(paths[0], transform, mode.device)

    print(f"[{mode.label}] Running model-only benchmark...")
    model_stats = summarize(
        benchmark_model_only(
            model,
            tensor,
            mode.device,
            mode.amp,
            args.warmup_images,
            count,
        )
    )

    print(f"[{mode.label}] Running end-to-end benchmark...")
    e2e_stats = summarize(
        benchmark_end_to_end(
            paths,
            transform,
            model,
            mode.device,
            mode.amp,
            args.warmup_images,
            count,
        )
    )

    del tensor
    del model
    if mode.device.type == "cuda":
        torch.cuda.empty_cache()

    return {"model": model_stats, "e2e": e2e_stats}


def run_command(command: list[str]) -> str:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
            env=env,
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


def gpu_info() -> tuple[str, str, str, str]:
    if not torch.cuda.is_available():
        return "N/A", "N/A", "N/A", "N/A"
    p = torch.cuda.get_device_properties(0)
    return (
        p.name,
        f"{p.total_memory / 1024**3:.2f} GiB",
        str(p.multi_processor_count),
        f"{p.major}.{p.minor}",
    )


def stat_row(mode_label: str, scope: str, stats: dict[str, float]) -> str:
    return (
        f"| {mode_label} | {scope} | {stats['fps']:.2f} | {stats['mean']:.3f} | "
        f"{stats['p50']:.3f} | {stats['p95']:.3f} | {stats['p99']:.3f} | "
        f"{stats['min']:.3f} | {stats['max']:.3f} |"
    )


def report_text(
    args: argparse.Namespace,
    modes: list[BenchmarkMode],
    paths: list[Path],
    count: int,
    results: dict[str, dict[str, dict[str, float]]],
    snapshots: dict[str, str],
) -> str:
    gpu_name, vram, sms, capability = gpu_info()
    nvidia_text = snapshots["nvidia-smi"]
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", nvidia_text)
    driver_cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", nvidia_text)
    driver = driver_match.group(1) if driver_match else "Unavailable"
    driver_cuda = driver_cuda_match.group(1) if driver_cuda_match else "Unavailable"

    lines = [
        "# Inference Speed Benchmark",
        "",
        f"Generated: `{datetime.now().astimezone().isoformat(timespec='seconds')}`",
        "",
        "This file was generated locally by `temp/infer_direct.py`. The script does not upload or commit this report to Git.",
        "",
        "## Results",
        "",
        "All measurements use `batch_size=1`, sequential requests, and exclude model/checkpoint loading time.",
        "",
        "| Mode | Scope | FPS | Mean ms | P50 ms | P95 ms | P99 ms | Min ms | Max ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for mode in modes:
        stats = results[mode.key]
        lines.append(stat_row(mode.label, "Model forward only", stats["model"]))
        lines.append(stat_row(mode.label, "End-to-end", stats["e2e"]))

    lines += [
        "",
        "- **Model forward only:** a preprocessed `[1, 3, H, W]` tensor already resides on the inference device; only `model(tensor)` is timed.",
        "- **End-to-end:** file open/decode → RGB → resize/crop/normalize → device transfer → model forward → argmax → scalar result on CPU.",
        "- CPU mode has no CPU→GPU transfer. GPU modes include the CPU→GPU transfer in end-to-end timing.",
        "- GPU AMP uses CUDA FP16 autocast; CPU mode is FP32.",
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
        f"| Benchmark modes | {', '.join(mode.label for mode in modes)} |",
        f"| Unique source images | {len(paths)} |",
        f"| Timed requests per mode/scope | {count} |",
        f"| Warmup requests per mode/scope | {args.warmup_images} |",
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
        f"| PyTorch CPU intra-op threads | {torch.get_num_threads()} |",
        f"| PyTorch CPU inter-op threads | {torch.get_num_interop_threads()} |",
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

    modes = resolve_modes(args.modes)
    paths = collect_image_paths(args.data_path)
    random.Random(args.seed).shuffle(paths)
    count = len(paths) if args.timed_images == 0 else args.timed_images
    transform = build_transform(args.input_size)

    results: dict[str, dict[str, dict[str, float]]] = {}
    for mode in modes:
        results[mode.key] = benchmark_mode(mode, args, paths, transform, count)

    snapshots = command_snapshots(args.data_path)
    report = report_text(args, modes, paths, count, results, snapshots)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")

    print("")
    print("Benchmark summary")
    for mode in modes:
        stats = results[mode.key]
        print(
            f"{mode.label}: model_forward_fps={stats['model']['fps']:.2f}, "
            f"end_to_end_fps={stats['e2e']['fps']:.2f}"
        )
    print(f"report: {output}")


if __name__ == "__main__":
    main()
