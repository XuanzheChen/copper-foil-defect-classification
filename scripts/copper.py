#!/usr/bin/env python3
"""Cross-platform entry point for the standalone Copper Foil classifier.

The implementation delegates to the original training/evaluation/inference code.
It does not modify the models, image transforms, or checkpoint format.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve().parent


def labels_from_file(path):
    labels = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        index, sep, name = line.strip().partition(":")
        if not sep or index != str(len(labels)) or not name.strip():
            raise ValueError("Labels must use contiguous index:class_name lines from 0")
        if "/" in name or "\\" in name or name in {".", ".."}:
            raise ValueError("Invalid class directory name in label file")
        labels.append(name)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Class labels must be nonempty and unique")
    return labels


def verify_split(folder, labels):
    if not folder.is_dir():
        raise ValueError(f"Dataset directory does not exist: {folder}")
    actual = sorted(p.name for p in folder.iterdir() if p.is_dir())
    if actual != sorted(labels):
        raise ValueError(
            f"Classes in {folder} do not match label file. "
            f"Expected: {sorted(labels)}; found: {actual}"
        )
    if actual != labels:
        raise ValueError(
            "ImageFolder class indices use lexicographic folder order. "
            "Reorder the label file to match this order: " + repr(actual)
        )


def resolve_device(requested):
    if requested == "cpu":
        return "cpu"
    import torch
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise ValueError("CUDA requested but PyTorch cannot access an NVIDIA GPU")
    return "cuda" if available else "cpu"


def launch(program, options, labels=None):
    cmd = [sys.executable, str(ROOT / program), *map(str, options)]
    if labels is not None:
        cmd.append(str(labels))
    print("Running:", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def checkpoint_settings(checkpoint):
    import torch
    # PyTorch checkpoint metadata may contain argparse.Namespace; load only
    # checkpoints from a trusted source. Never load unknown .pth files.
    stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(stored, dict):
        raise ValueError("Unsupported checkpoint: expected a dictionary")
    meta = stored.get("args")
    if isinstance(meta, dict):
        get = meta.get
    else:
        get = lambda key: getattr(meta, key, None)
    return get("model"), get("input_size"), get("nb_classes")


def run_train(args):
    labels = labels_from_file(args.labels)
    for part in ("train", "val"):
        verify_split(args.data / part, labels)
    device = resolve_device(args.device)
    if args.pretrained and not args.pretrained.is_file():
        raise ValueError(f"Pretrained weights not found: {args.pretrained}")
    if args.resume and not args.resume.is_file():
        raise ValueError(f"Resume checkpoint not found: {args.resume}")
    args.output.mkdir(parents=True, exist_ok=True)
    opts = [
        "--model", args.model, "--data_set", "IMNET",
        "--data_path", args.data, "--nb_classes", len(labels),
        "--input_size", 200, "--epochs", args.epochs,
        "--warmup_epochs", min(30, args.epochs),
        "--batch_size", args.batch_size, "--update_freq", args.update_freq,
        "--blr", args.blr, "--layer_decay_type", "single",
        "--layer_decay", 0.9, "--weight_decay", 0.05,
        "--drop_path", 0.2, "--mixup", 0.8, "--cutmix", 1.0,
        "--smoothing", 0.1, "--best_acc_tolerance", 0.10,
        "--save_ckpt_num", 1, "--use_amp", str(device == "cuda").lower(),
        "--HSVEnhance", "true", "--add_Gauss_Noise", "false",
        "--device", device, "--num_workers", args.workers,
        "--seed", args.seed, "--output_dir", args.output,
    ]
    if args.pretrained:
        opts += ["--finetune", args.pretrained]
    if args.resume:
        opts += ["--resume", args.resume, "--auto_resume", "false"]
    launch("main_finetune.py", opts, args.labels)


def run_eval(args):
    labels = labels_from_file(args.labels)
    verify_split(args.data / "train", labels)
    verify_split(args.data / args.split, labels)
    if not args.checkpoint.is_file():
        raise ValueError(f"Checkpoint missing: {args.checkpoint}")
    model, size, count = checkpoint_settings(args.checkpoint)
    if model is None or size is None or count is None:
        raise ValueError("Checkpoint lacks model/input_size/nb_classes metadata")
    if int(size) != 200 or int(count) != len(labels):
        raise ValueError("Checkpoint input size or class count does not match data")
    device = resolve_device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    opts = [
        "--model", model, "--eval", "true", "--data_set", "image_folder",
        "--data_path", args.data / "train",
        "--eval_data_path", args.data / args.split,
        "--nb_classes", len(labels), "--input_size", size,
        "--resume", args.checkpoint, "--auto_resume", "false",
        "--output_dir", args.output, "--batch_size", args.batch_size,
        "--num_workers", args.workers, "--device", device,
        "--use_amp", str(device == "cuda").lower(),
    ]
    launch("main_finetune.py", opts, args.labels)


def run_infer(args):
    labels = labels_from_file(args.labels)
    if not args.data.is_dir():
        raise ValueError("For the combined infer/organize workflow, --data must be a directory")
    if not args.checkpoint.is_file():
        raise ValueError(f"Checkpoint missing: {args.checkpoint}")
    if not 1 <= args.k <= len(labels):
        raise ValueError(f"--k must be between 1 and {len(labels)}")
    if args.output.exists():
        raise ValueError(f"Output directory already exists: {args.output}")
    device = resolve_device(args.device)
    # Snapshot before inference, so the downstream organizer can verify each
    # file and detect changes between inference and organization.
    args.output.mkdir(parents=True)
    launch("scripts/create-data-manifest.py", [
        "--data", args.data, "--output", args.output / "manifest.json"
    ])
    launch("infer_topk.py", [
        "--data", args.data, "--checkpoint", args.checkpoint,
        "--class-labels", args.labels, "--output", args.output,
        "--k", args.k, "--conf", args.conf,
        "--device", device, "--amp", str(device == "cuda").lower(),
        "--batch-size", args.batch_size, "--num-workers", args.workers,
        "--seed", args.seed,
    ])


def run_organize(args):
    if not args.data.is_dir():
        raise ValueError(f"Input directory missing: {args.data}")
    if not args.inference.is_dir():
        raise ValueError(f"Inference result directory missing: {args.inference}")
    launch("scripts/organize-inference-results.py", [
        "--data", args.data,
        "--predictions", args.inference / "all_predictions.jsonl",
        "--summary", args.inference / "summary.json",
        "--manifest", args.manifest or (args.inference / "manifest.json"),
        "--class-labels", args.labels,
        "--output", args.output,
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Supervised training")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--labels", type=Path, required=True)
    train.add_argument("--model", default="convnextv2_tiny")
    train.add_argument("--pretrained", type=Path)
    train.add_argument("--resume", type=Path)
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--update-freq", type=int, default=4)
    train.add_argument("--blr", type=float, default=3.2e-3)
    train.add_argument("--output", type=Path, required=True)

    evaluation = sub.add_parser("eval", help="Evaluate a trained checkpoint")
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--labels", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluation.add_argument("--batch-size", type=int, default=32)
    evaluation.add_argument("--output", type=Path, required=True)

    inference = sub.add_parser("infer", help="Infer a directory and write a snapshot manifest")
    inference.add_argument("--data", type=Path, required=True)
    inference.add_argument("--labels", type=Path, required=True)
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--k", type=int, default=3)
    inference.add_argument("--conf", type=float, default=0.7)
    inference.add_argument("--batch-size", type=int, default=32)
    inference.add_argument("--output", type=Path, required=True)

    organize = sub.add_parser("organize", help="Organize predictions into class folders")
    organize.add_argument("--data", type=Path, required=True)
    organize.add_argument("--labels", type=Path, required=True)
    organize.add_argument("--inference", type=Path, required=True)
    organize.add_argument("--manifest", type=Path)
    organize.add_argument("--output", type=Path, required=True)

    for cmd in (train, evaluation, inference):
        cmd.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        cmd.add_argument("--workers", type=int, default=0)
        cmd.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        for name in ("epochs", "batch_size", "update_freq"):
            if hasattr(args, name) and getattr(args, name) < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
        if hasattr(args, "workers") and args.workers < 0:
            raise ValueError("--workers must be nonnegative")
        if getattr(args, "command", None) == "train":
            run_train(args)
        elif args.command == "eval":
            run_eval(args)
        elif args.command == "infer":
            run_infer(args)
        else:
            run_organize(args)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode or 1, f"Command failed: {error}\n")


if __name__ == "__main__":
    main()
