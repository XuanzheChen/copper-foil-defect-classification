#!/usr/bin/env python3
"""Build a self-contained seven-class deployment artifact; never commit weights."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_package(repo, output, checkpoint, source_commit):
    repo, output, checkpoint = map(lambda p: Path(p).resolve(), (repo, output, checkpoint))
    template = repo / "deployment/minimal"
    info = json.loads((template / "model_info.json").read_text(encoding="utf-8"))
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if sha256(checkpoint) != info["checkpoint_sha256"]:
        raise ValueError("Checkpoint does not match the selected training run")
    sources = {name: repo / name for name in
               ("infer_topk.py", "datasets.py", "models/self/convnextv2_modified.py", "models/utils.py", "LICENSE")}
    sources.update({name: template / name for name in
                    ("README.md", "infer.sh", "requirements.txt", "model_info.json", "verify_package.py", "configs/class_labels_csd.txt")})
    # Canonical UTF-8/LF text yields identical Windows/Linux package bytes.
    content = {name: path.read_text(encoding="utf-8-sig").encode("utf-8") for name, path in sources.items()}
    # Preserve the original training label bytes, including their line endings.
    label_name = "configs/class_labels_csd.txt"
    candidates = (content[label_name], content[label_name].replace(b"\n", b"\r\n"))
    content[label_name] = next((data for data in candidates
                                if hashlib.sha256(data).hexdigest() == info["class_labels_sha256"]), content[label_name])
    labels = [line.split(":", 1)[1] for line in content["configs/class_labels_csd.txt"].decode().splitlines()]
    if len(labels) != info["num_classes"] or hashlib.sha256(content["configs/class_labels_csd.txt"]).hexdigest() != info["class_labels_sha256"]:
        raise ValueError("Labels do not match the selected training run")
    output.mkdir(parents=True)
    for name, data in content.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for name in ("models/__init__.py", "models/self/__init__.py"):
        (output / name).write_bytes(b"")
    weights = output / "weights/checkpoint-best.pth"
    weights.parent.mkdir()
    shutil.copyfile(checkpoint, weights)
    (output / "infer.sh").chmod(0o755)
    manifest = {
        "format_version": 1, "source_commit": source_commit, "training_run": info["training_run"],
        "checkpoint_sha256": info["checkpoint_sha256"], "class_labels_sha256": info["class_labels_sha256"],
        "classes": labels,
        "files": {p.relative_to(output).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
                  for p in sorted(output.rglob("*")) if p.is_file()},
    }
    (output / "PACKAGE_MANIFEST.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    spec = importlib.util.spec_from_file_location("package_verifier", output / "verify_package.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify(output)
    return manifest


def create_archive(package, archive):
    package, archive = Path(package), Path(archive)
    if archive.exists() or package.resolve() in archive.resolve().parents:
        raise ValueError("Archive must be a new file outside the package")
    archive.parent.mkdir(parents=True, exist_ok=True)
    # Stored bytes and fixed metadata make Windows/Linux ZIPs identical.
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as output:
        files = (p for p in package.rglob("*") if p.is_file() and "__pycache__" not in p.relative_to(package).parts)
        for path in sorted(files, key=lambda p: p.relative_to(package).as_posix()):
            relative = path.relative_to(package).as_posix()
            entry = zipfile.ZipInfo("copper-foil-7class/" + relative, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = (0o100755 if relative == "infer.sh" else 0o100644) << 16
            entry.file_size = path.stat().st_size
            with path.open("rb") as source, output.open(entry, "w") as target:
                shutil.copyfileobj(source, target, 1024 * 1024)
    return sha256(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    try:
        manifest = create_package(repo, args.output, args.checkpoint, commit)
        archive_sha = create_archive(args.output, args.zip) if args.zip else None
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({"package": str(args.output.resolve()), "files": len(manifest["files"]),
                      "checkpoint_sha256": manifest["checkpoint_sha256"], "zip_sha256": archive_sha}, ensure_ascii=False))


if __name__ == "__main__":
    main()
