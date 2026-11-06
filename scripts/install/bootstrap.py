#!/usr/bin/env python3
"""Install the standalone Copper repository without changing global Python.

A thin Bash/PowerShell launcher installs/locates uv and a managed Python 3.10
interpreter. This Python module handles safe Git operations, CUDA selection,
virtualenv setup, dependency installation, and smoke verification.

Run from a checked-out portable/standalone copy or with --target to clone into
another directory. HTTPS cloning of the public repository needs no SSH key;
SSH cloning remains optional with a properly configured GitHub key.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit

BRANCH = "portable/standalone"
REPO = "XuanzheChen/copper-foil-defect-classification"
REPO_URL = f"https://github.com/{REPO}.git"
PROFILE_FILENAME = Path("scripts/install/environment_profiles.json")
PYTHON_VERSION = (3, 10)


class InstallError(RuntimeError):
    """User-facing installation error, never silently downgraded."""


def run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
        capture: bool = False, dry_run: bool = False) -> str:
    safe_command = subprocess.list2cmdline([str(x) for x in cmd])
    print(f"$ {safe_command}", flush=True)
    if dry_run:
        return ""
    try:
        result = subprocess.run([str(x) for x in cmd], cwd=cwd, env=env,
                                check=True, text=True,
                                stdout=subprocess.PIPE if capture else None,
                                stderr=subprocess.PIPE if capture else None)
    except FileNotFoundError as err:
        raise InstallError(f"Missing executable: {cmd[0]}") from err
    except subprocess.CalledProcessError as err:
        message = (err.stderr or err.stdout or "").strip()
        raise InstallError(f"Command failed (exit {err.returncode}): {safe_command}\n{message}") from err
    return (result.stdout or "").strip()


def repo_origin_matches(origin: str) -> bool:
    """Accept only HTTPS/SSH URL for this repository, without embedded tokens."""
    origin = origin.strip().rstrip("/")
    if re.fullmatch(r"git@github\.com:XuanzheChen/copper-foil-defect-classification(?:\.git)?",
                    origin, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"ssh://git@github\.com/XuanzheChen/copper-foil-defect-classification(?:\.git)?",
                    origin, flags=re.IGNORECASE):
        return True
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return (parsed.scheme == "https" and parsed.hostname == "github.com" and
            parsed.username is None and parsed.password is None and
            parsed.port is None and parsed.query == "" and parsed.fragment == "" and
            parsed.path.rstrip("/").removesuffix(".git").casefold() == f"/{REPO}".casefold())


def git_text(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args], capture=True)


def prepare_repository(target: Path, *, update: bool, dry_run: bool,
                       repo_url: str = REPO_URL) -> bool:
    """Return True if the directory already contained the proper checkout."""
    if not repo_origin_matches(repo_url):
        raise InstallError("--repo-url must be an authorized GitHub HTTPS/SSH URL for the Copper repository; do not put tokens in URLs")
    if not target.exists() or (target.is_dir() and not any(target.iterdir())):
        run(["git", "clone", "--depth", "1", "--single-branch", "--branch",
             BRANCH, repo_url, str(target)], dry_run=dry_run)
        return False
    if target.is_symlink() or not target.is_dir():
        raise InstallError(f"Target is not a safe directory: {target}")
    if not (target / ".git").exists():
        raise InstallError(f"Nonempty target is not a Git repository: {target}. Refusing overwrite.")
    current_branch = git_text(target, "symbolic-ref", "--quiet", "--short", "HEAD")
    if current_branch != BRANCH:
        raise InstallError(f"Expected branch {BRANCH}, found {current_branch}. Refusing to change branches.")
    origin = git_text(target, "remote", "get-url", "origin")
    if not repo_origin_matches(origin):
        raise InstallError("Target Git origin does not match the expected Copper repository")
    if update:
        if git_text(target, "status", "--porcelain", "--untracked-files=normal"):
            raise InstallError("Working tree has uncommitted/untracked files; --update refuses to overwrite")
        # Do not shallow-fetch a single tip only: FF ancestry must be available.
        run(["git", "-C", str(target), "fetch", "origin", BRANCH], dry_run=dry_run)
        run(["git", "-C", str(target), "merge", "--ff-only", "FETCH_HEAD"], dry_run=dry_run)
    return True


def normalize_os() -> str:
    name = platform.system()
    if name not in {"Windows", "Linux"}:
        raise InstallError(f"Unsupported OS: {name}. Use Windows or Linux x86-64.")
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        raise InstallError(f"Unsupported CPU architecture: {platform.machine()} (requires x86-64)")
    if name == "Linux":
        libc_name, libc_ver = platform.libc_ver()
        if libc_name == "glibc" and libc_ver and parse_version(libc_ver) < parse_version("2.28"):
            raise InstallError(f"glibc {libc_ver} is too old for PyTorch 2.7.1 manylinux_2_28 wheels")
    return name


def parse_version(value: str) -> tuple[int, ...]:
    value = value.strip()
    if not re.fullmatch(r"\d+(?:\.\d+)*", value):
        raise InstallError(f"Invalid version string: {value!r}")
    return tuple(int(part) for part in value.split("."))


def version_at_least(actual: str, required: str) -> bool:
    a, b = parse_version(actual), parse_version(required)
    size = max(len(a), len(b))
    return a + (0,) * (size - len(a)) >= b + (0,) * (size - len(b))


def capability(value: str) -> str | None:
    match = re.fullmatch(r"\s*(\d+)[.,](\d+)\s*", value)
    return f"{int(match.group(1))}.{int(match.group(2))}" if match else None


def detect_gpu(gpu_index: int) -> dict[str, Any] | None:
    """Use nvidia-smi, never nvcc; a GPU can work without local CUDA Toolkit."""
    if shutil.which("nvidia-smi") is None:
        return None
    fields = ["index", "name", "driver_version", "compute_cap"]
    try:
        output = run(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                      "--format=csv,noheader,nounits"], capture=True)
        records = list(csv.reader(output.splitlines()))
        rows = [dict(zip(fields, [item.strip() for item in row])) for row in records
                if len(row) == len(fields)]
    except InstallError:
        # Old nvidia-smi may not expose compute_cap. Never guess from the name.
        fields = fields[:-1]
        output = run(["nvidia-smi", "--query-gpu=" + ",".join(fields),
                      "--format=csv,noheader,nounits"], capture=True)
        records = list(csv.reader(output.splitlines()))
        rows = [dict(zip(fields, [item.strip() for item in row])) for row in records
                if len(row) == len(fields)]
    for row in rows:
        if row.get("index") == str(gpu_index):
            return {"index": gpu_index, "name": row["name"],
                    "driver": row["driver_version"],
                    "compute_capability": capability(row.get("compute_cap", ""))}
    raise InstallError(f"nvidia-smi found, but GPU index {gpu_index} is unavailable. Discovered: {[r.get('index') for r in rows]}")


def choose_profile(requested: str, gpu: dict[str, Any] | None,
                   os_name: str, profiles: dict[str, Any]) -> str:
    available = profiles["profiles"]
    if requested not in {"auto", *available}:
        raise InstallError(f"Unknown backend {requested!r}")
    if requested == "cpu":
        return "cpu"
    if gpu is None:
        if requested != "auto":
            raise InstallError("A CUDA backend was requested but nvidia-smi/NVIDIA driver is unavailable")
        print("INFO: No working nvidia-smi detected; explicitly selecting CPU profile (no silent GPU downgrade).")
        return "cpu"
    if requested == "auto":
        cap = gpu["compute_capability"]
        if cap is None:
            raise InstallError("Could not detect GPU compute capability. Select --backend cu118/cu126/cu128 explicitly after checking compatibility.")
        candidate_names = ["cu128", "cu126", "cu118"]
    else:
        candidate_names = [requested]
    matches = []
    for name in candidate_names:
        profile = available[name]
        min_driver = profile["min_driver"][os_name]
        if not version_at_least(gpu["driver"], min_driver):
            if requested != "auto":
                raise InstallError(f"{name} requires NVIDIA driver >= {min_driver} on {os_name}; detected {gpu['driver']}")
            continue
        if gpu["compute_capability"] and gpu["compute_capability"] not in profile["compute_capabilities"]:
            if requested != "auto":
                raise InstallError(f"GPU compute capability {gpu['compute_capability']} is not validated for {name}")
            continue
        matches.append(name)
    if not matches:
        raise InstallError(
            f"No validated CUDA profile for {gpu['name']} (CC={gpu['compute_capability']}, driver={gpu['driver']}). "
            "Update the NVIDIA driver or use --backend cpu explicitly."
        )
    return matches[0]


def read_profiles(root: Path) -> dict[str, Any]:
    location = root / PROFILE_FILENAME
    if not location.is_file():
        raise InstallError(f"Missing installation profiles in {location}")
    value = json.loads(location.read_text(encoding="utf-8"))
    if value.get("schema") != 1 or value.get("python") != "3.10":
        raise InstallError("Unsupported installation profile schema/Python version")
    return value


def venv_python(venv: Path, os_name: str) -> Path:
    return venv / ("Scripts/python.exe" if os_name == "Windows" else "bin/python")


def uv_program() -> str:
    uv = os.environ.get("COPPER_UV") or shutil.which("uv")
    if not uv:
        raise InstallError("uv is missing. Use install.sh/install.ps1 to bootstrap uv, or set COPPER_UV")
    return uv


def plan_install(root: Path, *, backend: str, gpu: dict[str, Any] | None,
                 os_name: str, profiles: dict[str, Any]) -> dict[str, Any]:
    profile = profiles["profiles"][backend]
    return {"os": os_name, "architecture": platform.machine(),
            "target": str(root), "branch": BRANCH, "backend": backend,
            "driver": gpu["driver"] if gpu else None,
            "gpu": gpu["name"] if gpu else None,
            "compute_capability": gpu["compute_capability"] if gpu else None,
            "pytorch": profiles["torch"], "torchvision": profiles["torchvision"],
            "python": profiles["python"], "wheel_index": profile["index_url"],
            "virtualenv": str(root / ".venv")}


def create_or_check_venv(root: Path, env: Path, uv: str, os_name: str,
                         profile: str, *, recreate: bool) -> Path:
    marker = env / "copper-install-state.json"
    if env.exists():
        if env.is_symlink() or not env.is_dir() or not (env / "pyvenv.cfg").is_file():
            raise InstallError(f"Refusing to manipulate non-venv path: {env}")
        if not marker.exists():
            raise InstallError(f"Existing venv is not managed by Copper: {env}. Move it aside manually before installing.")
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("schema") != 1:
            raise InstallError("Unknown venv ownership marker; refusing to alter it")
        if not recreate:
            if state.get("profile") != profile:
                raise InstallError(f"Existing venv profile {state.get('profile')} != {profile}. Use --recreate-env.")
        else:
            shutil.rmtree(env)
    if not env.exists():
        run([uv, "venv", "--python", sys.executable, str(env)])
        marker.write_text(json.dumps({"schema": 1, "profile": profile, "state": "installing"}, indent=2) + "\n", encoding="utf-8")
    py = venv_python(env, os_name)
    if not py.is_file():
        raise InstallError(f"Virtualenv Python missing: {py}")
    return py


def verify_install(py: Path, backend: str, gpu_index: int,
                   expected_cuda: str | None) -> dict[str, Any]:
    script = r'''
import json, os
import torch, torchvision, timm, numpy, PIL
from models.self import convnextv2_modified
backend = os.environ["COPPER_VERIFY_BACKEND"]
expected = os.environ.get("COPPER_VERIFY_CUDA") or None
if torch.__version__.split("+")[0] != "2.7.1" or torchvision.__version__.split("+")[0] != "0.22.1":
    raise RuntimeError("Unexpected torch/torchvision version")
if backend != "cpu":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch cannot access selected GPU")
    if not torch.version.cuda or not torch.version.cuda.startswith(expected):
        raise RuntimeError(f"Expected CUDA runtime {expected}, got {torch.version.cuda}")
    device = "cuda"
    cap = torch.cuda.get_device_capability(0)
    arch = torch.cuda.get_arch_list()
    print("GPU:", torch.cuda.get_device_name(0), "capability:", cap, "compiled arches:", arch)
else:
    if torch.version.cuda:
        raise RuntimeError("Expected CPU-only torch wheel, got CUDA build")
    device = "cpu"
x = torch.ones((4,), device=device)
assert float(x.sum().item()) == 4.0
model = convnextv2_modified.convnextv2_tiny(num_classes=2).to(device).eval()
with torch.inference_mode():
    output = model(torch.zeros((1,3,200,200), device=device))
    if isinstance(output, dict):
        output = output["logits"]
    if tuple(output.shape) != (1,2) or not torch.isfinite(output).all():
        raise RuntimeError("Model forward smoke check failed")
print("COPPER_VERIFIED=" + json.dumps({
  "torch": torch.__version__, "torchvision": torchvision.__version__,
  "timm": timm.__version__, "cuda_runtime": torch.version.cuda,
  "model_forward": True, "device": device
}))
'''
    env = os.environ.copy()
    env["COPPER_VERIFY_BACKEND"] = backend
    env["COPPER_VERIFY_CUDA"] = expected_cuda or ""
    if backend != "cpu":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    output = run([str(py), "-c", script], cwd=py.parents[2], capture=True, env=env)
    prefix = "COPPER_VERIFIED="
    results = [line for line in output.splitlines() if line.startswith(prefix)]
    if not results:
        raise InstallError("Smoke test did not produce a verification report")
    return json.loads(results[-1][len(prefix):])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, help="Clone/update this path; defaults to the installer checkout")
    parser.add_argument("--repo-url", default=REPO_URL, help="Copper HTTPS or SSH GitHub origin; no embedded tokens")
    parser.add_argument("--backend", choices=["auto", "cpu", "cu118", "cu126", "cu128"], default="auto")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--update", action="store_true", help="Fetch and fast-forward only when the working tree is clean")
    parser.add_argument("--recreate-env", action="store_true", help="Delete only the installer-managed .venv and recreate")
    parser.add_argument("--dry-run", action="store_true", help="Show detected platform/profile and planned changes without installing")
    args = parser.parse_args(argv)
    try:
        if args.gpu_index < 0:
            raise InstallError("--gpu-index must be nonnegative")
        if sys.version_info[:2] != PYTHON_VERSION:
            raise InstallError("Installer requires Python 3.10. Use install.sh/install.ps1 to provision managed Python 3.10")
        os_name = normalize_os()
        source_root = Path(__file__).resolve().parents[2]
        target = (args.target or source_root).expanduser().resolve()
        if not repo_origin_matches(args.repo_url):
            raise InstallError("Invalid --repo-url: expected authorized GitHub SSH/HTTPS Copper origin")
        if args.dry_run:
            if target.exists() and (target / ".git").exists():
                # Verify branch and origin on the target; no modifications.
                prepare_repository(target, update=False, dry_run=True, repo_url=args.repo_url)
            elif target.exists() and any(target.iterdir()):
                raise InstallError("Target is not an empty directory or verified repo")
            else:
                print(f"PLAN: would clone {BRANCH} from GitHub to {target}")
            profiles_root = target if (target / PROFILE_FILENAME).exists() else source_root
        else:
            prepare_repository(target, update=args.update, dry_run=False, repo_url=args.repo_url)
            profiles_root = target
        profiles = read_profiles(profiles_root)
        gpu = detect_gpu(args.gpu_index) if args.backend != "cpu" else None
        backend = choose_profile(args.backend, gpu, os_name, profiles)
        plan = plan_install(target, backend=backend, gpu=gpu, os_name=os_name, profiles=profiles)
        print("Install plan:\n" + json.dumps(plan, indent=2, ensure_ascii=False))
        if args.dry_run:
            print("DRY RUN: no clone, environment creation, or dependency installation performed")
            return 0
        uv = uv_program()
        env = target / ".venv"
        py = create_or_check_venv(target, env, uv, os_name, backend, recreate=args.recreate_env)
        p = profiles["profiles"][backend]
        run([uv, "pip", "install", "--python", str(py),
             "--index-url", p["index_url"],
             f"torch=={profiles['torch']}", f"torchvision=={profiles['torchvision']}"])
        run([uv, "pip", "install", "--python", str(py), "-r", str(target / "requirements.txt")])
        print("Running tensor, dependency and 200x200 ConvNeXt model smoke checks...")
        details = verify_install(py, backend, args.gpu_index, p["cuda_runtime"])
        # Avoid persisting URLs/credentials. No tokens or original images in report.
        sha = git_text(target, "rev-parse", "HEAD")
        report = {**plan, "git_commit": sha, "verification": details,
                  "completed_at": datetime.now(timezone.utc).isoformat()}
        (env / "copper-install-state.json").write_text(json.dumps({
            "schema": 1, "profile": backend, "state": "verified", "git_commit": sha
        }, indent=2) + "\n", encoding="utf-8")
        (env / "copper-install-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("SUCCESS: standalone Copper environment is installed and verified.")
        print(f"Python executable: {py}")
        print(f"Report: {env / 'copper-install-report.json'}")
        print(f"Example: {py} {target / 'scripts/copper.py'} --help")
        return 0
    except (InstallError, OSError, ValueError, KeyError, json.JSONDecodeError) as err:
        parser.exit(2, f"ERROR: {err}\n")


if __name__ == "__main__":
    raise SystemExit(main())
