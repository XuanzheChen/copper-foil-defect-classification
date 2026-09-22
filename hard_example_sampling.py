"""Hard-example mining and weighted sampling utilities.

Hardness is defined only by model scores for individual samples. No source,
batch, filename-prefix, or group metadata is used.
"""

import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Sampler


HARD_KINDS = {
    "hard_copper",
    "boundary_copper",
    "hard_foreign_negative",
}


def _relative_sample_path(path, dataset_root):
    root = Path(dataset_root).resolve()
    sample = Path(path).resolve()
    try:
        return sample.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"prediction path {sample} is outside dataset root {root}"
        ) from exc


def build_hard_example_manifest(
    predictions_path,
    dataset_root,
    output_path,
    copper_boundary_margin_max=0.5,
    foreign_negative_margin_min=-0.5,
    copper_label="铜瘤",
    foreign_label="异物",
):
    """Select hard train examples from deterministic inference predictions.

    copper_logit_margin is defined as:
        z_copper - max_{j != copper} z_j

    Copper positives:
      * margin < 0 -> hard_copper
      * 0 <= margin <= copper_boundary_margin_max -> boundary_copper

    Foreign-object negatives:
      * margin >= foreign_negative_margin_min -> hard_foreign_negative

    The output uses paths relative to the train root and intentionally contains
    no source/group metadata.
    """
    predictions_path = Path(predictions_path)
    output_path = Path(output_path)

    required = {
        "path",
        "true_label",
        "pred_label",
        "copper_prob",
        "copper_logit_margin",
        "copper_rival_label",
        "copper_rival_prob",
    }
    selected = []
    counts = {
        "hard_copper": 0,
        "boundary_copper": 0,
        "hard_foreign_negative": 0,
    }
    total_rows = 0
    copper_rows = 0
    foreign_rows = 0

    with predictions_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{predictions_path} is missing required columns: {sorted(missing)}"
            )

        for row in reader:
            total_rows += 1
            true_label = row["true_label"]
            margin = float(row["copper_logit_margin"])
            kind = None

            if true_label == copper_label:
                copper_rows += 1
                if margin < 0.0:
                    kind = "hard_copper"
                elif margin <= copper_boundary_margin_max:
                    kind = "boundary_copper"
            elif true_label == foreign_label:
                foreign_rows += 1
                if margin >= foreign_negative_margin_min:
                    kind = "hard_foreign_negative"

            if kind is None:
                continue

            selected.append(
                {
                    "relative_path": _relative_sample_path(
                        row["path"], dataset_root
                    ),
                    "kind": kind,
                    "true_label": true_label,
                    "pred_label": row["pred_label"],
                    "copper_logit_margin": margin,
                    "copper_prob": float(row["copper_prob"]),
                    "copper_rival_label": row["copper_rival_label"],
                    "copper_rival_prob": float(row["copper_rival_prob"]),
                }
            )
            counts[kind] += 1

    selected.sort(
        key=lambda row: (
            row["kind"],
            row["copper_logit_margin"],
            row["relative_path"],
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "relative_path",
        "kind",
        "true_label",
        "pred_label",
        "copper_logit_margin",
        "copper_prob",
        "copper_rival_label",
        "copper_rival_prob",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    summary = {
        "predictions": str(predictions_path),
        "dataset_root": str(Path(dataset_root)),
        "output_manifest": str(output_path),
        "selection": {
            "copper_boundary_margin_max": copper_boundary_margin_max,
            "foreign_negative_margin_min": foreign_negative_margin_min,
            "copper_label": copper_label,
            "foreign_label": foreign_label,
        },
        "total_prediction_rows": total_rows,
        "copper_rows": copper_rows,
        "foreign_rows": foreign_rows,
        "selected_rows": len(selected),
        "selected_by_kind": counts,
    }
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary


def build_hard_example_weights(dataset, manifest_path, factor):
    """Build per-sample weights from a manifest.

    All samples start at weight 1.0. Manifest-selected hard examples receive
    factor. The dataset epoch length is not changed; a weighted sampler changes
    only sampling probabilities.
    """
    factor = float(factor)
    if factor < 1.0:
        raise ValueError(f"hard example factor must be >= 1.0, got {factor}")

    weights = torch.ones(len(dataset), dtype=torch.double)
    if factor == 1.0:
        return weights, {
            "factor": factor,
            "selected": 0,
            "selected_by_kind": {},
            "manifest": str(manifest_path or ""),
        }

    if not manifest_path:
        raise ValueError("hard example factor > 1 requires --hard_example_manifest")

    root = Path(dataset.root).resolve()
    index_by_relative_path = {}
    for index, (sample_path, _) in enumerate(dataset.samples):
        relative = Path(sample_path).resolve().relative_to(root).as_posix()
        index_by_relative_path[relative] = index

    selected_indices = set()
    selected_by_kind = {}
    unmatched = []
    with Path(manifest_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"relative_path", "kind"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{manifest_path} is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            kind = row["kind"]
            if kind not in HARD_KINDS:
                raise ValueError(f"unknown hard-example kind: {kind}")
            relative = Path(row["relative_path"]).as_posix()
            index = index_by_relative_path.get(relative)
            if index is None:
                unmatched.append(relative)
                continue
            selected_indices.add(index)
            selected_by_kind[kind] = selected_by_kind.get(kind, 0) + 1

    if unmatched:
        preview = ", ".join(unmatched[:5])
        raise ValueError(
            f"{len(unmatched)} manifest samples were not found in the training "
            f"dataset; first entries: {preview}"
        )
    if not selected_indices:
        raise ValueError("hard-example manifest selected zero training samples")

    for index in selected_indices:
        weights[index] = factor

    return weights, {
        "factor": factor,
        "selected": len(selected_indices),
        "selected_by_kind": selected_by_kind,
        "manifest": str(manifest_path),
    }


class WeightedDistributedSampler(Sampler):
    """Fixed-length weighted sampler compatible with distributed training."""

    def __init__(
        self,
        weights,
        num_replicas=1,
        rank=0,
        seed=0,
        replacement=True,
    ):
        if num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must satisfy 0 <= rank < num_replicas")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        if self.weights.ndim != 1 or self.weights.numel() == 0:
            raise ValueError("weights must be a non-empty 1-D tensor")
        if torch.any(self.weights <= 0):
            raise ValueError("all sample weights must be > 0")

        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.replacement = bool(replacement)
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        return iter(indices[self.rank:self.total_size:self.num_replicas])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
