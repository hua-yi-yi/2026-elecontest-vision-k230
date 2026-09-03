#!/usr/bin/env python3
"""Build a deterministic PTQ set without using dataset_70596 test images."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "training" / "data" / "dataset_70596"
HARD_NEGATIVES = ROOT / "training" / "data" / "k230_hard_examples" / "images"
OUTPUT = ROOT / "tmp" / "dataset_70596_416_calibration"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def image_paths(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in EXTENSIONS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    sources = (
        ("train", DATASET / "images" / "train"),
        ("val", DATASET / "images" / "val"),
        ("negative", HARD_NEGATIVES),
    )
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)

    files = []
    counts = {}
    for prefix, folder in sources:
        paths = image_paths(folder)
        if not paths:
            raise FileNotFoundError("no calibration images in %s" % folder)
        counts[prefix] = len(paths)
        for index, source in enumerate(paths):
            destination = OUTPUT / ("%s_%03d%s" % (prefix, index, source.suffix.lower()))
            shutil.copy2(source, destination)
            files.append(
                {
                    "file": destination.name,
                    "source": source.resolve().as_posix(),
                    "sha256": sha256(destination),
                }
            )

    manifest = {
        "counts": counts,
        "total": len(files),
        "test_images_included": False,
        "files": files,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("DATASET_70596_CALIBRATION_READY", json.dumps({**counts, "total": len(files)}))
    print("output=%s" % OUTPUT)


if __name__ == "__main__":
    main()
