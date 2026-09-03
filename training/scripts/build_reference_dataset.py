#!/usr/bin/env python3
"""Make a reproducible K230 training split from the project field dataset.

The annotated images (237 real captures and 3,000 copy-paste field scenes)
are stored in ``datasets/field_scenes/``; this script builds the training split as file
lists instead of duplicating several hundred megabytes of images.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "datasets" / "field_scenes"
HARD = ROOT / "training" / "data" / "k230_hard_examples"
OUTPUT = ROOT / "training" / "data" / "reference_v1"


def bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100


def label_path(image: Path) -> Path:
    return image.parent.parent / "labels" / (image.stem + ".txt")


def checked_images(folder: Path) -> list[Path]:
    images = sorted((folder / "images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError("no images in %s" % folder)
    for image in images:
        label = label_path(image)
        if not label.is_file():
            raise FileNotFoundError("missing label for %s" % image)
    return images


def write_list(path: Path, images: list[Path]) -> None:
    path.write_text("\n".join(image.resolve().as_posix() for image in images) + "\n", encoding="utf-8")


def main() -> None:
    real = checked_images(REFERENCE / "real_field_sample")
    synth = checked_images(REFERENCE / "synth_field")
    hard_images = sorted((HARD / "images").glob("*.png")) + sorted((HARD / "images").glob("*.jpg"))
    if len(hard_images) < 2:
        raise FileNotFoundError("expected the K230 panel and dark-scene negatives in %s" % (HARD / "images"))
    for image in hard_images:
        label = HARD / "labels" / (image.stem + ".txt")
        if not label.is_file() or label.read_text(encoding="utf-8").strip():
            raise ValueError("hard negative must have an empty label: %s" % image)

    splits = {"train": [], "val": [], "test": []}
    # Only real field images enter test: this is the closest available measure
    # of real-world generalisation. Synthetic images supplement train/val.
    for image in real:
        value = bucket("real/" + image.name)
        splits["train" if value < 70 else "val" if value < 85 else "test"].append(image)
    for image in synth:
        value = bucket("synth/" + image.name)
        splits["train" if value < 88 else "val"].append(image)
    splits["train"].extend(hard_images)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, images in splits.items():
        write_list(OUTPUT / (name + ".txt"), images)
    yaml = OUTPUT / "steel_ball_reference.yaml"
    yaml.write_text(
        "path: %s\ntrain: train.txt\nval: val.txt\ntest: test.txt\n\nnames:\n  0: steel_ball\n" % OUTPUT.resolve().as_posix(),
        encoding="utf-8",
    )
    stats = {name: len(images) for name, images in splits.items()}
    print("REFERENCE_DATASET_READY", stats, yaml)


if __name__ == "__main__":
    main()
