"""Tests for model-score hard-example mining and sampling."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from hard_example_sampling import (
    WeightedDistributedSampler,
    build_hard_example_manifest,
    build_hard_example_weights,
)


class HardExampleSamplingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.root = self.tmp / "train"
        for cls in ("异物", "铜瘤", "其他"):
            (self.root / cls).mkdir(parents=True)

        self.samples = []
        for cls, name in (
            ("铜瘤", "hard.png"),
            ("铜瘤", "boundary.png"),
            ("铜瘤", "easy.png"),
            ("异物", "hardneg.png"),
            ("异物", "easyneg.png"),
            ("其他", "other.png"),
        ):
            path = self.root / cls / name
            path.touch()
            target = {"异物": 0, "铜瘤": 1, "其他": 2}[cls]
            self.samples.append((str(path), target))

        self.dataset = SimpleNamespace(
            root=str(self.root),
            samples=self.samples,
            __len__=lambda self_: len(self.samples),
        )

    def _write_predictions(self):
        path = self.tmp / "predictions.csv"
        fieldnames = [
            "path", "true_label", "pred_label", "copper_prob",
            "copper_logit_margin", "copper_rival_label", "copper_rival_prob",
        ]
        rows = [
            [self.samples[0][0], "铜瘤", "异物", 0.20, -1.0, "异物", 0.70],
            [self.samples[1][0], "铜瘤", "铜瘤", 0.55, 0.20, "异物", 0.40],
            [self.samples[2][0], "铜瘤", "铜瘤", 0.95, 3.00, "异物", 0.02],
            [self.samples[3][0], "异物", "异物", 0.35, -0.20, "异物", 0.50],
            [self.samples[4][0], "异物", "异物", 0.02, -3.00, "异物", 0.90],
            [self.samples[5][0], "其他", "其他", 0.30, -0.10, "其他", 0.40],
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fieldnames)
            writer.writerows(rows)
        return path

    def test_manifest_selects_only_score_defined_hard_examples(self):
        predictions = self._write_predictions()
        manifest = self.tmp / "hard_examples.csv"
        summary = build_hard_example_manifest(
            predictions, self.root, manifest,
            copper_boundary_margin_max=0.5,
            foreign_negative_margin_min=-0.5,
        )
        self.assertEqual(summary["selected_rows"], 3)
        self.assertEqual(summary["selected_by_kind"]["hard_copper"], 1)
        self.assertEqual(summary["selected_by_kind"]["boundary_copper"], 1)
        self.assertEqual(summary["selected_by_kind"]["hard_foreign_negative"], 1)

        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertNotIn("group", rows[0])
        self.assertNotIn("source", rows[0])
        selected_paths = {row["relative_path"] for row in rows}
        self.assertEqual(
            selected_paths,
            {"铜瘤/hard.png", "铜瘤/boundary.png", "异物/hardneg.png"},
        )

    def test_manifest_weights_only_selected_examples(self):
        predictions = self._write_predictions()
        manifest = self.tmp / "hard_examples.csv"
        build_hard_example_manifest(predictions, self.root, manifest)
        weights, info = build_hard_example_weights(self.dataset, manifest, 4.0)
        self.assertEqual(info["selected"], 3)
        self.assertEqual(weights.tolist(), [4.0, 4.0, 1.0, 4.0, 1.0, 1.0])

    def test_factor_one_returns_unit_weights_without_manifest_dependency(self):
        weights, info = build_hard_example_weights(self.dataset, "", 1.0)
        self.assertTrue(torch.equal(weights, torch.ones(6, dtype=torch.double)))
        self.assertEqual(info["selected"], 0)

    def test_weighted_sampler_is_fixed_length_reproducible_and_epoch_sensitive(self):
        weights = torch.tensor([1.0, 1.0, 4.0, 1.0, 4.0, 1.0], dtype=torch.double)
        sampler = WeightedDistributedSampler(weights, seed=7)
        epoch0_a = list(iter(sampler))
        epoch0_b = list(iter(sampler))
        self.assertEqual(len(epoch0_a), len(weights))
        self.assertEqual(epoch0_a, epoch0_b)
        sampler.set_epoch(1)
        epoch1 = list(iter(sampler))
        self.assertEqual(len(epoch1), len(weights))
        self.assertNotEqual(epoch0_a, epoch1)


if __name__ == "__main__":
    unittest.main()
