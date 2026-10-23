#!/usr/bin/env python3
"""Inference only: emit Top-k candidates when Top-1 softmax confidence < conf.

Copy this file beside datasets.py and models/ in the minimum deployment package.
No training entry point, optimizer, augmentation, or checkpoint fallback is used.
"""

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def integer_at_least(minimum):
    def parse(value):
        try:
            result = int(value)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError("must be an integer") from None
        if result < minimum:
            raise argparse.ArgumentTypeError(f"must be >= {minimum}")
        return result
    return parse


def confidence(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("conf must be a finite number in [0, 1]") from None
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("conf must be a finite number in [0, 1]")
    return result


def validate_options(k, conf, num_classes):
    if type(k) is not int or not 1 <= k <= num_classes:
        raise ValueError(f"k must be an integer in [1, {num_classes}]")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        raise ValueError("conf must be a finite number in [0, 1]")
    if not math.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError("conf must be a finite number in [0, 1]")


def read_labels(path):
    labels = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        index, separator, name = line.strip().partition(":")
        if not separator or index != str(len(labels)) or not name.strip():
            raise ValueError("labels must be sequential index:name lines starting at 0")
        labels.append(name.strip())
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("labels must contain unique, nonempty class names")
    return labels


def make_prediction(probabilities, labels, k, conf, ranked_indices=None):
    """Keep original probabilities; equality with conf does not trigger Top-k."""
    validate_options(k, conf, len(labels))
    if len(probabilities) != len(labels) or any(
        not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities
    ):
        raise ValueError("invalid probability vector")
    if ranked_indices is None:
        ranked_indices = sorted(range(len(labels)), key=lambda i: (-probabilities[i], i))
    if sorted(ranked_indices) != list(range(len(labels))):
        raise ValueError("ranked_indices must be a permutation of class indices")
    first = ranked_indices[0]
    low = probabilities[first] < conf
    return {
        "pred_label": labels[first],
        "top1_conf": probabilities[first],
        "low_confidence": low,
        "topk": [
            {"class_id": i, "class_name": labels[i], "confidence": probabilities[i]}
            for i in ranked_indices[:k]
        ] if low else [],
    }


class InferenceImages:
    """A map-style inference dataset supporting labeled folders or plain images."""

    def __init__(self, root, labels):
        from PIL import Image
        from datasets import build_transform

        root = Path(root).resolve()
        extensions = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
        paths = [root] if root.is_file() else sorted(
            (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions),
            key=lambda p: p.relative_to(root).as_posix(),
        )
        self.samples = []
        self.excluded = []
        self.transform = build_transform(False, SimpleNamespace(input_size=200))
        for path in paths:
            with Image.open(path) as image:
                width, height = image.size
            if (width, height) != (200, 200):
                self.excluded.append({"path": str(path), "width": width, "height": height})
                continue
            parts = path.relative_to(root).parts if root.is_dir() else ()
            truth = parts[0] if len(parts) > 1 and parts[0] in labels else None
            self.samples.append((str(path), truth))
        if not self.samples:
            raise ValueError("no admissible 200x200 images in --data")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from datasets import load_grayscale_image
        return self.transform(load_grayscale_image(self.samples[index][0])), index


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(records, labels, k, conf, elapsed, excluded):
    labeled = [r for r in records if r["true_label"] is not None]
    errors = [r for r in labeled if not r["correct"]]
    low = [r for r in records if r["low_confidence"]]
    low_labeled = [r for r in low if r["true_label"] is not None]
    report = []
    for label in labels:
        support = sum(r["true_label"] == label for r in labeled)
        predicted = sum(r["pred_label"] == label for r in labeled)
        tp = sum(r["true_label"] == r["pred_label"] == label for r in labeled)
        report.append({
            "class": label, "support": support, "correct": tp, "predicted": predicted,
            "recall": tp / support if support else None,
            "precision": tp / predicted if predicted else None,
        })
    return {
        "k": k, "conf": conf, "comparison": "top1_conf < conf (strict)",
        "samples": len(records), "excluded_non_200x200": excluded,
        "labeled_samples": len(labeled), "correct": len(labeled) - len(errors),
        "errors": len(errors), "accuracy": 1 - len(errors) / len(labeled) if labeled else None,
        "low_confidence_samples": len(low),
        "low_confidence_correct": sum(r["correct"] for r in low_labeled),
        "low_confidence_errors": sum(not r["correct"] for r in low_labeled),
        "low_confidence_topk_contains_truth": sum(
            any(c["class_name"] == r["true_label"] for c in r["topk"]) for r in low_labeled
        ),
        "per_class": report, "inference_seconds": elapsed,
        "images_per_second": len(records) / elapsed,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="ImageFolder, unlabeled folder, or single image")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--class-labels", type=Path, default=Path(__file__).resolve().parent / "configs/class_labels_csd.txt")
    parser.add_argument("--k", required=True, type=integer_at_least(1))
    parser.add_argument("--conf", required=True, type=confidence)
    parser.add_argument("--batch-size", default=32, type=integer_at_least(1))
    parser.add_argument("--num-workers", default=4, type=integer_at_least(0))
    parser.add_argument("--seed", default=0, type=integer_at_least(0))
    parser.add_argument("--model", default=None, help="default: architecture stored in checkpoint")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--amp", default="true", choices=("true", "false"))
    return parser


def run(args, labels):
    import numpy as np
    import torch
    from models.self import convnextv2_modified as architectures

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; choose --device cpu explicitly")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dataset = InferenceImages(args.data, labels)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stored = checkpoint.get("args")
    stored_model = getattr(stored, "model", None)
    name = args.model or stored_model or "convnextv2_tiny"
    if args.model and stored_model and args.model != stored_model:
        raise ValueError("--model differs from the checkpoint architecture")
    if getattr(stored, "nb_classes", len(labels)) != len(labels):
        raise ValueError("checkpoint class count differs from --class-labels")
    factory = getattr(architectures, name, None)
    if factory is None or not name.startswith("convnext"):
        raise ValueError(f"unsupported model: {name}")
    model = factory(num_classes=len(labels))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    del checkpoint
    use_amp = device.type == "cuda" and args.amp == "true"
    records = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for images, indices in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext():
                logits = model(images)
                if isinstance(logits, dict):
                    logits = logits["logits"]
            if not torch.isfinite(logits).all():
                raise ValueError("model produced nonfinite logits")
            # Match the reference evaluator's softmax dtype. Do not round before filtering.
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()
            order = torch.argsort(logits, dim=1, descending=True, stable=True).cpu().tolist()
            for index, probs, ranked in zip(indices.tolist(), probabilities, order):
                path, truth = dataset.samples[index]
                result = make_prediction(probs, labels, args.k, args.conf, ranked)
                result.update(index=index, path=path, true_label=truth,
                              correct=(truth == result["pred_label"]) if truth is not None else None)
                records.append(result)
            if device.type == "cuda":
                torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    summary = summarize(records, labels, args.k, args.conf, elapsed, len(dataset.excluded))
    summary.update(model=name, checkpoint=str(args.checkpoint.resolve()),
                   checkpoint_sha256=file_sha256(args.checkpoint),
                   class_labels_sha256=file_sha256(args.class_labels),
                   data=str(args.data.resolve()), seed=args.seed, batch_size=args.batch_size,
                   num_workers=args.num_workers, device=str(device), amp=use_amp,
                   probability_dtype="float16" if use_amp else "float32",
                   strict_weights_loaded=True)
    args.output.mkdir(parents=True, exist_ok=True)
    low = [r for r in records if r["low_confidence"]]
    for filename, rows in (("all_predictions.jsonl", records), ("low_confidence_topk.jsonl", low)):
        with (args.output / filename).open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    fields = ["index", "path", "true_label", "correct"] + [
        f"top{rank}_{field}" for rank in range(1, args.k + 1) for field in ("label", "conf")
    ]
    with (args.output / "low_confidence_topk.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in low:
            flattened = {key: row[key] for key in fields[:4]}
            for rank, candidate in enumerate(row["topk"], 1):
                flattened[f"top{rank}_label"] = candidate["class_name"]
                flattened[f"top{rank}_conf"] = candidate["confidence"]
            writer.writerow(flattened)
    with (args.output / "excluded_non_200x200.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "width", "height"])
        writer.writeheader()
        writer.writerows(dataset.excluded)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    for row in low:
        print(json.dumps(row, ensure_ascii=False, allow_nan=False))
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), file=sys.stderr)
    return summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        labels = read_labels(args.class_labels)
        validate_options(args.k, args.conf, len(labels))
        if not args.data.exists():
            raise ValueError("--data does not exist")
        if not args.checkpoint.is_file():
            raise ValueError("--checkpoint does not exist")
        run(args, labels)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
