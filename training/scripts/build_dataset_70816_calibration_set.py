#!/usr/bin/env python3
"""Build a deterministic PTQ set for dataset_70816 without test leakage."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "training" / "data" / "dataset_70816_merged"
OUTPUT = ROOT / "tmp" / "dataset_70816_416_calibration"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
TRAIN_SAMPLES = 72
VAL_SAMPLES = 24


def image_paths(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in EXTENSIONS)


def evenly_spaced(paths: list[Path], count: int) -> list[Path]:
    if len(paths) < count:
        raise ValueError("need %d images, found %d" % (count, len(paths)))
    if count == 1:
        return [paths[len(paths) // 2]]
    return [paths[round(index * (len(paths) - 1) / (count - 1))] for index in range(count)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    train = image_paths(DATASET / "images" / "train")
    validation = image_paths(DATASET / "images" / "val")
    test_hashes = {sha256(path) for path in image_paths(DATASET / "images" / "test")}
    negative_names = {
        "k230_dark_scene_negative.png",
        "k230_perforated_panel_negative.png",
    }
    negatives = [path for path in train if path.name in negative_names]
    positives = [path for path in train if path.name not in negative_names]
    if len(negatives) != len(negative_names):
        raise ValueError("missing hard-negative calibration images")

    selected = [("train", path) for path in evenly_spaced(positives, TRAIN_SAMPLES)]
    selected.extend(("val", path) for path in evenly_spaced(validation, VAL_SAMPLES))
    selected.extend(("negative", path) for path in negatives)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    files = []
    counts = {"train": 0, "val": 0, "negative": 0}
    for index, (source_type, source) in enumerate(selected):
        digest = sha256(source)
        if digest in test_hashes:
            raise ValueError("test image leaked into calibration: %s" % source)
        destination = OUTPUT / ("%s_%03d%s" % (source_type, index, source.suffix.lower()))
        shutil.copy2(source, destination)
        counts[source_type] += 1
        files.append(
            {
                "file": destination.name,
                "source": source.resolve().as_posix(),
                "sha256": digest,
            }
        )

    manifest = {
        "counts": counts,
        "total": len(files),
        "selection": "evenly spaced over sorted train and val images plus all hard negatives",
        "test_images_included": False,
        "files": files,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("DATASET_70816_CALIBRATION_READY", json.dumps({**counts, "total": len(files)}))
    print("output=%s" % OUTPUT)


if __name__ == "__main__":
    main()
