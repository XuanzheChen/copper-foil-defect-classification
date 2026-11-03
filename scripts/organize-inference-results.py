#!/usr/bin/env python3
"""Organize inference images by Top-1 while preserving originals and provenance.

Portable across Windows and Linux. Supports arbitrary class counts, Top-k and
confidence thresholds from the inference summary. No PyTorch dependency.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.as_posix() != value:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path.as_posix()


def recorded_path(value):
    if not isinstance(value, str):
        raise ValueError("Path recorded in inference results must be a string")
    if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("\\\\"):
        return PureWindowsPath(value)
    return PurePosixPath(value.replace("\\", "/"))


def finite_probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def parse_labels(path):
    labels = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        index, separator, label = line.partition(":")
        if separator != ":" or index != str(len(labels)) or not label:
            raise ValueError("Invalid sequential class labels")
        if relative_path(label) != label or "/" in label or any(c in label for c in '<>:"|?*'):
            raise ValueError("Invalid class directory name")
        if label.rstrip(". ") != label:
            raise ValueError("Invalid class directory name for Windows")
        labels.append(label)
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Expected unique, nonempty class labels")
    return labels


def organize(data, predictions, summary_path, manifest_path, labels_path, output):
    data, output = Path(data).resolve(), Path(output).resolve()
    if data == output or data in output.parents:
        raise ValueError("Output cannot be inside the source dataset")
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    labels = parse_labels(labels_path)
    k, conf = summary["k"], summary["conf"]
    if type(k) is not int or not 1 <= k <= len(labels) or not finite_probability(conf):
        raise ValueError("Invalid k/conf in inference summary")

    files = {}
    for record in manifest["files"]:
        name = relative_path(record["path"])
        if name in files:
            raise ValueError(f"Duplicate manifest path: {name}")
        source = data / name
        if source.is_symlink() or not source.is_file() or data not in source.resolve().parents:
            raise ValueError(f"Invalid source image: {name}")
        if source.stat().st_size != record["bytes"] or sha256(source) != record["sha256"]:
            raise ValueError(f"Source differs from uploaded snapshot: {name}")
        files[name] = record
    if any(p.is_symlink() for p in data.rglob("*")):
        raise ValueError("Source dataset contains symlinks")
    actual = {p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_file()}
    if actual != set(files) or len(files) != manifest["total_files"]:
        raise ValueError("Input file set differs from upload manifest")

    expected = {
        name for name, record in files.items()
        if record["is_image"] and (record["width"], record["height"]) == (200, 200)
    }
    excluded = sum(record["is_image"] for record in files.values()) - len(expected)
    source_root = recorded_path(summary["data"])
    rows, seen = [], set()
    for line in Path(predictions).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            name = relative_path(
                recorded_path(row["path"]).relative_to(source_root).as_posix()
            )
            if row.get("relative_path") is not None and relative_path(row["relative_path"]) != name:
                raise ValueError("relative_path does not match source path")
        except (ValueError, TypeError):
            raise ValueError("Prediction path is outside recorded input root") from None
        if name not in expected or name in seen:
            raise ValueError(f"Unexpected or duplicate prediction: {name}")
        seen.add(name)
        probability, label = row["top1_conf"], row["pred_label"]
        if label not in labels or not finite_probability(probability) or type(row["low_confidence"]) is not bool:
            raise ValueError(f"Invalid Top-1 prediction: {name}")
        low = probability < conf
        if row["low_confidence"] != low or len(row["topk"]) != (k if low else 0):
            raise ValueError(f"Invalid strict threshold/Top-{k} output: {name}")
        if low:
            candidates = row["topk"]
            if (candidates[0]["class_name"], candidates[0]["confidence"]) != (label, probability):
                raise ValueError(f"Top-{k} first candidate differs from Top-1")
            ids, previous = set(), 1.0
            for candidate in candidates:
                idx, value = candidate["class_id"], candidate["confidence"]
                if type(idx) is not int or not 0 <= idx < len(labels) or candidate["class_name"] != labels[idx]:
                    raise ValueError(f"Invalid Top-{k} class")
                if not finite_probability(value) or value > previous or idx in ids:
                    raise ValueError(f"Invalid Top-{k} probabilities/order")
                ids.add(idx)
                previous = value
        rows.append((name, row))

    if seen != expected or len(rows) != summary["samples"] or len(expected) != manifest["admissible_images"]:
        raise ValueError("Predictions do not cover all admissible images exactly once")
    if excluded != summary["excluded_non_200x200"]:
        raise ValueError("Excluded image count differs from manifest")
    if sum(row["low_confidence"] for _, row in rows) != summary["low_confidence_samples"]:
        raise ValueError("Low-confidence count differs from inference summary")

    low_filename = f"low_confidence_top{k}.json"
    names = ("by_top1", "top1_predictions.csv", low_filename, "organization_summary.json")
    if any((output / name).exists() for name in names):
        raise ValueError("Refusing to overwrite existing organized results")

    for label in labels:
        (output / "by_top1" / label).mkdir(parents=True, exist_ok=False)
    counts = dict.fromkeys(labels, 0)
    low_counts = dict.fromkeys(labels, 0)
    low_rows = []
    with (output / "top1_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "source_path", "organized_path", "top1_label", "top1_conf", "low_confidence", "sha256"
        ])
        writer.writeheader()
        for name, row in rows:
            label = row["pred_label"]
            relative = (PurePosixPath("by_top1") / label / name).as_posix()
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data / name, destination)
            if sha256(destination) != files[name]["sha256"]:
                raise ValueError(f"Copied image hash mismatch: {name}")
            counts[label] += 1
            record = {
                "source_path": name, "organized_path": relative,
                "top1_label": label, "top1_conf": row["top1_conf"],
                "low_confidence": row["low_confidence"], "sha256": files[name]["sha256"],
            }
            writer.writerow(record)
            if row["low_confidence"]:
                low_counts[label] += 1
                low_rows.append({**record, f"top{k}": row["topk"]})
    document = {
        "schema_version": 1, "source_root": str(source_root), "conf": conf, "k": k,
        "comparison": "top1_conf < conf (strict)", "samples": len(low_rows), "records": low_rows,
    }
    (output / low_filename).write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    report = {
        "source_files": len(files), "source_images": sum(r["is_image"] for r in files.values()),
        "classified_images": len(rows), "excluded_non_200x200": excluded,
        "non_image_files": sum(not r["is_image"] for r in files.values()),
        "per_class": counts, "low_confidence_per_class": low_counts,
        "low_confidence_samples": len(low_rows), "conf": conf, "k": k,
        "original_images_preserved": True, "all_copied_image_hashes_verified": True,
        "manifest_sha256": sha256(manifest_path), "predictions_sha256": sha256(predictions),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "class_labels_sha256": summary["class_labels_sha256"],
    }
    (output / "organization_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "predictions", "summary", "manifest", "class-labels", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        result = organize(
            args.data, args.predictions, args.summary, args.manifest,
            args.class_labels, args.output,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
