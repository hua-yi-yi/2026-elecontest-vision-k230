#!/usr/bin/env python3
"""Build a deterministic, label-free PTQ calibration set for K230 exports."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ("real", ROOT / "datasets" / "field_scenes" / "real_field_sample" / "images", 50),
    ("synth", ROOT / "datasets" / "field_scenes" / "synth_field" / "images", 50),
    ("k230neg", ROOT / "training" / "data" / "k230_hard_examples" / "images", 99),
    ("k230pos", ROOT / "training" / "data" / "k230_hard_examples" / "holdout_unlabelled", 99),
)
OUTPUT = ROOT / "tmp" / "k230_416_calibration"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    count = 0
    for prefix, source_dir, limit in SOURCES:
        images = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in EXTENSIONS)[:limit]
        if not images:
            raise FileNotFoundError("no calibration images in %s" % source_dir)
        for index, source in enumerate(images):
            destination = OUTPUT / ("%s_%03d%s" % (prefix, index, source.suffix.lower()))
            shutil.copy2(source, destination)
            count += 1
    print("CALIBRATION_SET_READY", count, OUTPUT)


if __name__ == "__main__":
    main()
