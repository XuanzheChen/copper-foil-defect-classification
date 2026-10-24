#!/usr/bin/env python3
"""Check delivered files without importing PyTorch or running inference."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    expected = set(manifest["files"]) | {"PACKAGE_MANIFEST.json"}
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
              and "__pycache__" not in p.relative_to(root).parts}
    if actual != expected:
        raise ValueError(f"Package file set differs: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
    for name, record in manifest["files"].items():
        path = (root / name).resolve()
        if root not in path.parents or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Package file failed integrity check: {name}")
    if sha256(root / "weights/checkpoint-best.pth") != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint provenance mismatch")
    if sha256(root / "configs/class_labels_csd.txt") != manifest["class_labels_sha256"]:
        raise ValueError("Class-label provenance mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        manifest = verify(args.root)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps({"verified": True, "files": len(manifest["files"]),
                      "checkpoint_sha256": manifest["checkpoint_sha256"],
                      "classes": manifest["classes"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
