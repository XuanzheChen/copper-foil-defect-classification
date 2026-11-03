"""Standalone CLI and cross-platform dataset-organizer regression tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


organizer = load_module("scripts/organize-inference-results.py", "portable_organizer")
cli = load_module("scripts/copper.py", "portable_cli")


class PortableWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / "source"
        self.data.mkdir()
        self.image = self.data / "sample.png"
        self.image.write_bytes(b"snapshot-test")
        self.labels = self.root / "labels.txt"
        self.labels.write_text("0:Alpha\n1:Beta\n", encoding="utf-8")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "total_files": 1, "admissible_images": 1, "files": [{
                "path": "sample.png",
                "bytes": len(self.image.read_bytes()),
                "sha256": hashlib.sha256(self.image.read_bytes()).hexdigest(),
                "is_image": True, "width": 200, "height": 200,
            }],
        }), encoding="utf-8")
        self.predictions = self.root / "all_predictions.jsonl"
        self.summary = self.root / "summary.json"
        self.output = self.root / "organized"

    def write_predictions(self, root, record_path):
        self.summary.write_text(json.dumps({
            "data": root, "k": 2, "conf": 0.4, "samples": 1,
            "excluded_non_200x200": 0, "low_confidence_samples": 1,
            "checkpoint_sha256": "test", "class_labels_sha256": "labels",
        }), encoding="utf-8")
        row = {
            "path": record_path, "relative_path": "sample.png",
            "pred_label": "Alpha", "top1_conf": 0.35, "low_confidence": True,
            "topk": [
                {"class_id": 0, "class_name": "Alpha", "confidence": 0.35},
                {"class_id": 1, "class_name": "Beta", "confidence": 0.34},
            ],
        }
        self.predictions.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def organize(self):
        return organizer.organize(
            self.data, self.predictions, self.summary, self.manifest,
            self.labels, self.output,
        )

    def test_dynamic_k_conf_linux_paths(self):
        self.write_predictions("/srv/copper/data", "/srv/copper/data/sample.png")
        result = self.organize()
        self.assertEqual(result["k"], 2)
        self.assertEqual(result["conf"], 0.4)
        self.assertTrue((self.output / "low_confidence_top2.json").exists())
        self.assertEqual((self.output / "by_top1/Alpha/sample.png").read_bytes(), self.image.read_bytes())

    def test_windows_recorded_paths_on_any_os(self):
        self.write_predictions("C:\\copper\\images", "C:\\copper\\images\\sample.png")
        result = self.organize()
        self.assertEqual(result["classified_images"], 1)
        self.assertTrue(self.image.exists())

    def test_reject_mismatched_snapshot_before_copy(self):
        self.write_predictions("/srv/data", "/srv/data/sample.png")
        self.image.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            self.organize()
        self.assertFalse(self.output.exists())

    def test_class_order_validation(self):
        self.assertEqual(cli.labels_from_file(self.labels), ["Alpha", "Beta"])
        folder = self.root / "split"
        (folder / "Alpha").mkdir(parents=True)
        (folder / "Beta").mkdir(parents=True)
        cli.verify_split(folder, ["Alpha", "Beta"])
        with self.assertRaisesRegex(ValueError, "Reorder"):
            cli.verify_split(folder, ["Beta", "Alpha"])


if __name__ == "__main__":
    unittest.main()
