"""Boundary and inference-output contract tests; no GPU or weights required."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from infer_topk import make_prediction, read_labels, summarize, validate_options


class TopKContractTests(unittest.TestCase):
    def test_integer_and_class_count_boundaries(self):
        for bad in (0, -1, 8, 3.0, 2.5, True, "3"):
            with self.subTest(k=bad), self.assertRaises(ValueError):
                validate_options(bad, 0.7, 7)
        validate_options(1, 0, 7)
        validate_options(7, 1, 7)

    def test_confidence_rejects_nonfinite_and_out_of_range(self):
        for bad in (-0.01, 1.01, float("nan"), float("inf"), -float("inf"), True, "0.7"):
            with self.subTest(conf=bad), self.assertRaises(ValueError):
                validate_options(3, bad, 7)

    def test_threshold_is_strict_and_probabilities_are_not_rounded(self):
        labels = ["凹凸", "压坑", "铜瘤"]
        equal = make_prediction([0.7, 0.2, 0.1], labels, 3, 0.7)
        self.assertFalse(equal["low_confidence"])
        self.assertEqual(equal["topk"], [])
        below = make_prediction([0.6999999, 0.2, 0.1000001], labels, 3, 0.7)
        self.assertTrue(below["low_confidence"])
        self.assertEqual([p["class_name"] for p in below["topk"]], labels)
        self.assertEqual(below["topk"][0]["confidence"], 0.6999999)
        self.assertFalse(make_prediction([1.0, 0.0, 0.0], labels, 1, 1)["low_confidence"])
        self.assertFalse(make_prediction([0.4, 0.3, 0.3], labels, 1, 0)["low_confidence"])

    def test_all_classes_can_be_requested_and_logits_break_probability_ties(self):
        labels = [str(i) for i in range(7)]
        result = make_prediction([0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1], labels, 7, 1,
                                 [1, 0, 2, 3, 4, 5, 6])
        self.assertEqual(result["pred_label"], "1")
        self.assertEqual(len(result["topk"]), 7)

    def test_invalid_probability_vectors_fail_closed(self):
        for vector in ([0.5], [float("nan"), 0.5], [1.1, -0.1]):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                make_prediction(vector, ["A", "B"], 1, 0.7)

    def test_cli_rejects_invalid_arguments_before_opening_weights_or_output(self):
        script = Path(__file__).resolve().parents[1] / "infer_topk.py"
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / "labels.txt"
            labels.write_text("".join(f"{i}:class{i}\n" for i in range(7)), encoding="utf-8")
            base = [sys.executable, str(script), "--class-labels", str(labels),
                    "--data", tmp, "--checkpoint", str(Path(tmp) / "missing.pth"),
                    "--output", str(Path(tmp) / "out")]
            for k, conf, expected in (("3.0", "0.7", "integer"), ("0", "0.7", ">= 1"),
                                      ("8", "0.7", "[1, 7]"), ("3", "nan", "finite"),
                                      ("3", "inf", "finite"), ("3", "1.1", "[0, 1]"),
                                      ("3", "-0.1", "[0, 1]")):
                with self.subTest(k=k, conf=conf):
                    result = subprocess.run(base + ["--k", k, "--conf", conf],
                                            text=True, capture_output=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(expected, result.stderr)
                    self.assertNotIn("missing.pth", result.stderr.split("error:")[-1])
                    self.assertFalse((Path(tmp) / "out").exists())

    def test_label_indices_and_duplicates_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.txt"
            for content in ("1:A\n", "0:A\n1:A\n", "0:A\n2:B\n", ""):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_labels(path)

    def test_metrics_distinguish_flagged_correct_predictions_from_caught_errors(self):
        rows = [dict(true_label="A", pred_label="A", correct=True, low_confidence=False, topk=[]),
                dict(true_label="B", pred_label="A", correct=False, low_confidence=True,
                     topk=[dict(class_name="A"), dict(class_name="B")]),
                dict(true_label=None, pred_label="A", correct=None, low_confidence=True, topk=[])]
        result = summarize(rows, ["A", "B"], 2, 0.7, 1.0, 1)
        self.assertEqual(result["labeled_samples"], 2)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["low_confidence_samples"], 2)
        self.assertEqual(result["low_confidence_errors"], 1)
        self.assertEqual(result["low_confidence_topk_contains_truth"], 1)


if __name__ == "__main__":
    unittest.main()
