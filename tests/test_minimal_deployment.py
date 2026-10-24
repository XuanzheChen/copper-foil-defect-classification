"""Portable package assembly, provenance, tampering, and deterministic ZIP tests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_minimal", REPO / "scripts/build-minimal-deployment.py")
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


class MinimalDeploymentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.repo = self.base / "source"
        template = self.repo / "deployment/minimal"
        shutil.copytree(REPO / "deployment/minimal", template)
        for name in ("infer_topk.py", "datasets.py", "models/self/convnextv2_modified.py", "models/utils.py", "LICENSE"):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"example source\r\n")
        self.checkpoint = self.base / "test-checkpoint.pth"
        self.checkpoint.write_bytes(b"unit test checkpoint bytes")
        info = json.loads((template / "model_info.json").read_text(encoding="utf-8"))
        info["checkpoint_sha256"] = builder.sha256(self.checkpoint)
        (template / "model_info.json").write_text(json.dumps(info), encoding="utf-8")

    def build(self, name):
        path = self.base / name
        manifest = builder.create_package(self.repo, path, self.checkpoint, "test-commit")
        return path, manifest

    def test_bundle_contains_matching_weights_labels_and_normalized_text(self):
        package, manifest = self.build("package")
        self.assertEqual(len(manifest["classes"]), 7)
        self.assertEqual((package / "weights/checkpoint-best.pth").read_bytes(), self.checkpoint.read_bytes())
        self.assertEqual((package / "infer_topk.py").read_bytes(), b"example source\n")
        self.assertNotIn("main_finetune.py", manifest["files"])

    def test_wrong_checkpoint_is_rejected_before_creating_output(self):
        self.checkpoint.write_bytes(b"wrong checkpoint")
        with self.assertRaisesRegex(ValueError, "Checkpoint"):
            self.build("bad")
        self.assertFalse((self.base / "bad").exists())

    def test_existing_output_is_preserved(self):
        package, _ = self.build("package")
        before = (package / "PACKAGE_MANIFEST.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.build("package")
        self.assertEqual((package / "PACKAGE_MANIFEST.json").read_bytes(), before)

    def test_archive_is_reproducible_and_self_contained(self):
        first, _ = self.build("first")
        second, _ = self.build("second")
        a, b = self.base / "a.zip", self.base / "b.zip"
        self.assertEqual(builder.create_archive(first, a), builder.create_archive(second, b))
        with zipfile.ZipFile(a) as archive:
            self.assertIn("copper-foil-7class/weights/checkpoint-best.pth", archive.namelist())
            self.assertEqual(archive.namelist(), sorted(archive.namelist()))
            self.assertEqual(archive.testzip(), None)

    def test_integrity_verifier_rejects_modified_file(self):
        package, _ = self.build("package")
        spec = importlib.util.spec_from_file_location("verify_test", package / "verify_package.py")
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        (package / "configs/class_labels_csd.txt").write_bytes(b"wrong labels")
        with self.assertRaisesRegex(ValueError, "integrity"):
            verifier.verify(package)


if __name__ == "__main__":
    unittest.main()
