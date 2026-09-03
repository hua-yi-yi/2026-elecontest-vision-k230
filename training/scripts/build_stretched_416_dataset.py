#!/usr/bin/env python3
"""Resize a YOLO dataset to physical 416x416 images for K230 direct-resize parity."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "training" / "data" / "dataset_70816_merged"
DEFAULT_OUTPUT = ROOT / "training" / "data" / "dataset_70816_merged_stretched_416"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not (args.source / "dataset.yaml").is_file():
        raise SystemExit("source dataset missing: %s" % args.source)
    if args.output.exists():
        if not args.force:
            raise SystemExit("output already exists; pass --force to rebuild: %s" % args.output)
        shutil.rmtree(args.output)

    stats = {}
    for split in ("train", "val", "test"):
        source_images = sorted(
            path for path in (args.source / "images" / split).iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        destination_images = args.output / "images" / split
        destination_labels = args.output / "labels" / split
        destination_images.mkdir(parents=True)
        destination_labels.mkdir(parents=True)
        source_sizes = {}
        for source in source_images:
            destination = destination_images / (source.stem + ".jpg")
            with Image.open(source) as image:
                source_sizes["%dx%d" % image.size] = source_sizes.get("%dx%d" % image.size, 0) + 1
                image.convert("RGB").resize(
                    (args.size, args.size), Image.Resampling.BILINEAR
                ).save(destination, quality=95, subsampling=0)
            source_label = args.source / "labels" / split / (source.stem + ".txt")
            if not source_label.is_file():
                raise ValueError("missing label: %s" % source_label)
            shutil.copy2(source_label, destination_labels / source_label.name)
        stats[split] = {"images": len(source_images), "source_sizes": source_sizes}

    yaml_path = args.output / "dataset.yaml"
    yaml_path.write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: steel_ball\n"
        % args.output.resolve().as_posix(),
        encoding="utf-8",
    )
    manifest = {
        "source": args.source.resolve().as_posix(),
        "output_size": [args.size, args.size],
        "resize": "direct bilinear stretch matching K230 AI2D preprocess",
        "labels": "normalized YOLO coordinates unchanged by axis-wise stretch",
        "stats": stats,
        "test_images_used_for_training": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("STRETCHED_DATASET_READY", json.dumps(manifest))
    print("yaml=%s" % yaml_path)


if __name__ == "__main__":
    main()
