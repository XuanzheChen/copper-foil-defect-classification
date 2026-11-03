#!/usr/bin/env python3
"""Build a portable, content-addressed manifest for an inference dataset."""
import argparse
import hashlib
import json
from pathlib import Path

EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(data):
    from PIL import Image
    data = Path(data).resolve()
    if not data.is_dir():
        raise ValueError(f"Input must be a directory: {data}")
    files = []
    admissible = 0
    for path in sorted(data.rglob("*"), key=lambda p: p.relative_to(data).as_posix()):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(data).as_posix()
        is_image = path.suffix.lower() in EXTENSIONS
        item = {
            "path": rel, "bytes": path.stat().st_size,
            "sha256": sha256(path), "is_image": is_image,
        }
        if is_image:
            with Image.open(path) as image:
                item["width"], item["height"] = image.size
            admissible += (item["width"], item["height"]) == (200, 200)
        files.append(item)
    if not admissible:
        raise ValueError("No admissible 200x200 image in the input directory")
    return {
        "format_version": 1,
        "total_files": len(files),
        "admissible_images": admissible,
        "files": files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        data, out = args.data.resolve(), args.output.resolve()
        if data == out or data in out.parents:
            raise ValueError("Manifest must be stored outside the input directory")
        if out.exists():
            raise ValueError(f"Refusing to overwrite existing manifest: {out}")
        document = build_manifest(data)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Snapshot: {document['total_files']} files, {document['admissible_images']} eligible images")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
