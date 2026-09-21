"""Tests for business-aware checkpoint selection."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_finetune import _should_update_best_checkpoint


class CheckpointSelectionTests(unittest.TestCase):
    def test_higher_copper_recall_wins_inside_accuracy_band(self):
        selected = {
            "acc1": 98.25,
            "macro_recall": 97.50,
            "copper_recall": 95.0,
            "copper_precision": 99.5,
        }
        candidate = {
            "acc1": 98.20,
            "macro_recall": 97.45,
            "copper_recall": 97.0,
            "copper_precision": 99.0,
        }
        self.assertTrue(
            _should_update_best_checkpoint(candidate, selected, 98.25, 0.10)
        )

    def test_accuracy_band_is_anchored_to_global_max_and_cannot_drift(self):
        selected = {
            "acc1": 98.18,
            "macro_recall": 97.50,
            "copper_recall": 98.0,
            "copper_precision": 99.0,
        }
        candidate = {
            "acc1": 98.14,
            "macro_recall": 98.00,
            "copper_recall": 100.0,
            "copper_precision": 100.0,
        }
        self.assertFalse(
            _should_update_best_checkpoint(candidate, selected, 98.25, 0.10)
        )

    def test_new_accuracy_max_forces_replacement_if_old_selection_leaves_band(self):
        selected = {
            "acc1": 98.10,
            "macro_recall": 98.00,
            "copper_recall": 100.0,
            "copper_precision": 100.0,
        }
        candidate = {
            "acc1": 98.40,
            "macro_recall": 97.00,
            "copper_recall": 95.0,
            "copper_precision": 99.0,
        }
        self.assertTrue(
            _should_update_best_checkpoint(candidate, selected, 98.40, 0.10)
        )


if __name__ == "__main__":
    unittest.main()
