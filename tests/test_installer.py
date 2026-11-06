"""Offline tests for installer policy and safety. No GitHub/GPU/network required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("copper_bootstrap", ROOT / "scripts/install/bootstrap.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
PROFILES = json.loads((ROOT / "scripts/install/environment_profiles.json").read_text())


class ProfileTests(unittest.TestCase):
    def gpu(self, cap, driver="572.61"):
        return {"index": 0, "name": "Synthetic NVIDIA GPU", "compute_capability": cap, "driver": driver}

    def test_blackwell_requires_cu128(self):
        self.assertEqual(mod.choose_profile("auto", self.gpu("12.0"), "Linux", PROFILES), "cu128")
        with self.assertRaisesRegex(mod.InstallError, "No validated CUDA profile"):
            mod.choose_profile("auto", self.gpu("12.0", "560.28"), "Linux", PROFILES)

    def test_3050_laptop_old_driver_selects_cu118(self):
        self.assertEqual(mod.choose_profile("auto", self.gpu("8.6", "535.54"), "Linux", PROFILES), "cu118")

    def test_3050_new_driver_selects_cu128(self):
        self.assertEqual(mod.choose_profile("auto", self.gpu("8.6", "572.61"), "Windows", PROFILES), "cu128")

    def test_legacy_gpu_rejects_cu128(self):
        self.assertEqual(mod.choose_profile("auto", self.gpu("6.1", "572.61"), "Linux", PROFILES), "cu126")
        with self.assertRaisesRegex(mod.InstallError, "not validated"):
            mod.choose_profile("cu128", self.gpu("6.1"), "Linux", PROFILES)

    def test_gpu_detected_must_not_silently_fallback_cpu(self):
        with self.assertRaisesRegex(mod.InstallError, "No validated CUDA profile"):
            mod.choose_profile("auto", self.gpu("8.6", "390.0"), "Windows", PROFILES)
        self.assertEqual(mod.choose_profile("cpu", self.gpu("12.0"), "Linux", PROFILES), "cpu")

    def test_missing_gpu_cpu_selection(self):
        self.assertEqual(mod.choose_profile("auto", None, "Linux", PROFILES), "cpu")
        with self.assertRaisesRegex(mod.InstallError, "unavailable"):
            mod.choose_profile("cu126", None, "Linux", PROFILES)

    def test_unknown_compute_capability_requires_explicit_profile(self):
        with self.assertRaisesRegex(mod.InstallError, "Could not detect GPU compute capability"):
            mod.choose_profile("auto", self.gpu(None), "Windows", PROFILES)
        self.assertEqual(mod.choose_profile("cu118", self.gpu(None, "535.54"), "Windows", PROFILES), "cu118")

    def test_driver_thresholds_platform_specific(self):
        self.assertTrue(mod.version_at_least("570.65", "570.65"))
        self.assertFalse(mod.version_at_least("570.64", "570.65"))
        self.assertTrue(mod.version_at_least("560.28.03", "560.28"))
        self.assertFalse(mod.version_at_least("560.28.02", "560.28.03"))
        with self.assertRaisesRegex(mod.InstallError, "requires NVIDIA driver"):
            mod.choose_profile("cu128", self.gpu("12.0", "570.64"), "Windows", PROFILES)

    def test_public_repository_uses_https_by_default(self):
        self.assertEqual(
            mod.REPO_URL,
            "https://github.com/XuanzheChen/copper-foil-defect-classification.git",
        )
        self.assertTrue(mod.repo_origin_matches(mod.REPO_URL))

    def test_origin_never_accepts_embedded_tokens_or_other_repos(self):
        for url in (
            "git@github.com:XuanzheChen/copper-foil-defect-classification.git",
            "https://github.com/XuanzheChen/copper-foil-defect-classification.git",
            "ssh://git@github.com/XuanzheChen/copper-foil-defect-classification.git",
        ):
            self.assertTrue(mod.repo_origin_matches(url), url)
        for url in (
            "https://token@github.com/XuanzheChen/copper-foil-defect-classification.git",
            "https://github.com/XuanzheChen/other.git",
            "http://github.com/XuanzheChen/copper-foil-defect-classification",
            "/tmp/some-repo",
        ):
            self.assertFalse(mod.repo_origin_matches(url), url)

    def test_compute_capability_parser(self):
        self.assertEqual(mod.capability("8.6"), "8.6")
        self.assertEqual(mod.capability(" 12.0 "), "12.0")
        self.assertIsNone(mod.capability("Not Supported"))


class GitSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_new_clone_dry_run_no_side_effects(self):
        target = self.root / "new"
        with mock.patch.object(mod, "run", return_value="") as call:
            self.assertFalse(mod.prepare_repository(target, update=False, dry_run=True))
            self.assertFalse(target.exists())
            self.assertIn("clone", call.call_args.args[0])
            self.assertTrue(call.call_args.kwargs["dry_run"])

    def test_nonempty_nongit_dir_refused(self):
        target = self.root / "already-has-data"
        target.mkdir()
        (target / "valuable.txt").write_text("KEEP")
        with self.assertRaisesRegex(mod.InstallError, "Nonempty"):
            mod.prepare_repository(target, update=True, dry_run=False)
        self.assertEqual((target / "valuable.txt").read_text(), "KEEP")

    def test_existing_git_wrong_branch_rejected(self):
        target = self.root / "repo"
        (target / ".git").mkdir(parents=True)
        with mock.patch.object(mod, "git_text", side_effect=["main"]):
            with self.assertRaisesRegex(mod.InstallError, "Expected branch"):
                mod.prepare_repository(target, update=True, dry_run=False)

    def test_existing_git_dirty_update_rejected(self):
        target = self.root / "repo"
        (target / ".git").mkdir(parents=True)
        with mock.patch.object(mod, "git_text", side_effect=[mod.BRANCH, mod.REPO_URL, " M script.py"]):
            with mock.patch.object(mod, "run") as call:
                with self.assertRaisesRegex(mod.InstallError, "Working tree"):
                    mod.prepare_repository(target, update=True, dry_run=False)
                call.assert_not_called()

    def test_explicit_update_only_fast_forwards(self):
        target = self.root / "repo"
        (target / ".git").mkdir(parents=True)
        with mock.patch.object(mod, "git_text", side_effect=[mod.BRANCH, mod.REPO_URL, ""]):
            with mock.patch.object(mod, "run") as call:
                mod.prepare_repository(target, update=True, dry_run=False)
                self.assertEqual(call.call_count, 2)
                self.assertEqual(call.call_args_list[1].args[0][-2:], ["--ff-only", "FETCH_HEAD"])

    def test_venv_requires_copper_marker(self):
        env = self.root / ".venv"
        env.mkdir()
        (env / "pyvenv.cfg").write_text("test")
        for recreate in (False, True):
            with self.assertRaisesRegex(mod.InstallError, "not managed"):
                mod.create_or_check_venv(self.root, env, "uv", "Linux", "cpu", recreate=recreate)

    def test_existing_venv_profile_mismatch_refused(self):
        env = self.root / ".venv"
        env.mkdir()
        (env / "pyvenv.cfg").write_text("test")
        (env / "copper-install-state.json").write_text(json.dumps({"schema": 1, "profile": "cu126"}))
        with self.assertRaisesRegex(mod.InstallError, "profile"):
            mod.create_or_check_venv(self.root, env, "uv", "Linux", "cpu", recreate=False)


if __name__ == "__main__":
    unittest.main()
