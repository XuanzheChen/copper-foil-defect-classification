"""Tests for the evaluation diagnostics added to engine_finetune.evaluate.

Covers the confusion-matrix and per-sample prediction artifacts: the CSV files
are written with the expected shape, each prediction traces back to a real file
path, and the confidence/margin invariant (margin = true-prob - max-prob:
~0 on correct, < 0 when the true class lost to the prediction) holds.

engine_finetune imports timm and tensorboardX transitively, so this file runs in
the training environment rather than against a bare interpreter.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torchvision import transforms

from datasets import CopperFoilImageFolder, build_transform
from engine_finetune import _classification_report_from_confusion, evaluate


class _FakeLinearModel(torch.nn.Module):
    """Small CNN-free stand-in that emits real logits for a tiny dataset.

    Needs >= 5 classes so evaluate()'s unconditional accuracy(topk=(1, 5))
    does not hit torch.topk's out-of-range check.
    """

    def __init__(self, num_classes=5):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 200 * 200, num_classes)

    def forward(self, x):
        return self.linear(x.flatten(1))


def _write_sample(path, value):
    Image.new("L", (200, 200), value).save(path)


CLASSES = ("class_a", "class_b", "class_c", "class_d", "class_e")


def _build_dataloader(tmp):
    root = Path(tmp) / "data"
    for cls_name, value in zip(CLASSES, (60, 160, 20, 120, 200)):
        (root / cls_name).mkdir(parents=True)
        for i in range(4):
            _write_sample(root / cls_name / f"{i}.png", value)

    args = SimpleNamespace(input_size=200)
    dataset = CopperFoilImageFolder(
        root=root,
        transform=build_transform(False, args),
        exclusion_report_path=Path(tmp) / "excluded.csv",
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=0
    )
    return dataset, loader


class EvalArtifactTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.dataset, self.loader = _build_dataloader(self.tmp)

    def _run_eval(self, **kwargs):
        model = _FakeLinearModel(num_classes=5)
        return evaluate(
            self.loader,
            model,
            "cpu",
            output_dir=str(self.tmp),
            save_predictions=True,
            save_confusion=True,
            class_names=list(CLASSES),
            num_classes=5,
            **kwargs,
        )

    def test_confusion_and_prediction_artifacts_are_written(self):
        self._run_eval()

        counts_path = self.tmp / "confusion_matrix.csv"
        normalized_path = self.tmp / "confusion_matrix_normalized.csv"
        predictions_path = self.tmp / "eval_predictions.csv"
        misclassified_path = self.tmp / "eval_misclassified.csv"

        for path in (counts_path, normalized_path, predictions_path, misclassified_path):
            self.assertTrue(path.is_file(), msg=f"missing {path.name}")

        with counts_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["true_label"] + list(CLASSES))
        self.assertEqual(len(rows), 1 + len(CLASSES))
        counts = [int(v) for row in rows[1:] for v in row[1:]]
        self.assertEqual(sum(counts), len(self.dataset))

        with predictions_path.open(newline="", encoding="utf-8") as handle:
            prediction_rows = list(csv.DictReader(handle))
        self.assertEqual(len(prediction_rows), len(self.dataset))
        expected_columns = {
            "index", "path", "true_label", "pred_label", "true_prob",
            "pred_prob", "margin", "correct", "top5",
        }
        self.assertEqual(set(prediction_rows[0].keys()), expected_columns)
        for row in prediction_rows:
            self.assertTrue(Path(row["path"]).is_file(), msg=row["path"])
            self.assertIn(row["true_label"], set(CLASSES))

    def test_margin_invariant_and_misclassified_subset(self):
        self._run_eval()

        with (self.tmp / "eval_predictions.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))

        off_diagonal = 0
        for row in rows:
            margin = float(row["margin"])
            correct = int(row["correct"])
            if correct:
                self.assertAlmostEqual(margin, 0.0, places=6)
            else:
                self.assertLess(margin, 0.0)
                off_diagonal += 1

        with (self.tmp / "eval_misclassified.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            misclassified = list(csv.DictReader(handle))
        self.assertEqual(len(misclassified), off_diagonal)
        self.assertTrue(all(int(r["correct"]) == 0 for r in misclassified))

    def test_partial_tail_batch_yields_one_row_per_sample(self):
        """A non-divisible dataset exercises the short final batch the same way
        the real eval set does (2913 samples, batch 64 -> last batch of 33)."""
        root = self.tmp / "tail"
        for cls_name, value in zip(CLASSES, (60, 160, 20, 120, 200)):
            (root / cls_name).mkdir(parents=True)
            for i in range(2):
                _write_sample(root / cls_name / f"{i}.png", value)

        args = SimpleNamespace(input_size=200)
        dataset = CopperFoilImageFolder(
            root=root,
            transform=build_transform(False, args),
            exclusion_report_path=self.tmp / "excluded.csv",
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=4, shuffle=False, num_workers=0
        )
        self.assertEqual(len(dataset), 10)
        evaluate(
            loader,
            _FakeLinearModel(num_classes=5),
            "cpu",
            output_dir=str(self.tmp),
            save_predictions=True,
            save_confusion=True,
            class_names=list(CLASSES),
            num_classes=5,
        )
        with (self.tmp / "eval_predictions.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), len(dataset))
        # Reconstructed global index is exactly the dataset position.
        self.assertEqual([int(r["index"]) for r in rows], list(range(len(dataset))))

    def test_summary_metrics_include_macro_and_copper_precision_recall(self):
        confusion = torch.tensor(
            [
                [8, 2, 0, 0, 0],
                [0, 10, 0, 0, 0],
                [0, 0, 5, 0, 0],
                [0, 0, 0, 4, 1],
                [0, 2, 0, 0, 8],
            ],
            dtype=torch.long,
        )
        report = _classification_report_from_confusion(
            confusion,
            ["class_a", "class_b", "class_c", "class_d", "铜瘤"],
        )
        self.assertAlmostEqual(report["macro_recall"], 88.0, places=6)
        self.assertAlmostEqual(report["copper_recall"], 80.0, places=6)
        self.assertAlmostEqual(report["copper_precision"], 8 / 9 * 100.0, places=6)

    def test_artifacts_are_skipped_when_flags_are_off(self):
        model = _FakeLinearModel(num_classes=5)
        evaluate(
            self.loader,
            model,
            "cpu",
            output_dir=str(self.tmp),
            save_predictions=False,
            save_confusion=False,
            class_names=list(CLASSES),
            num_classes=5,
        )
        for name in (
            "confusion_matrix.csv",
            "confusion_matrix_normalized.csv",
            "eval_predictions.csv",
            "eval_misclassified.csv",
        ):
            self.assertFalse((self.tmp / name).exists(), msg=name)


if __name__ == "__main__":
    unittest.main()
