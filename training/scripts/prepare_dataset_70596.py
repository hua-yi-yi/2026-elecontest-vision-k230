#!/usr/bin/env python3
"""Convert dataset_70596 from Pascal VOC XML to a reproducible YOLO split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = Path("/mnt/c/Users/31806/Downloads/dataset_70596.zip")
DEFAULT_OUTPUT = ROOT / "training" / "data" / "dataset_70596"
HARD_NEGATIVES = ROOT / "training" / "data" / "k230_hard_examples"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def bucket(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100


def split_for(filename: str) -> str:
    value = bucket("real/" + filename)
    return "train" if value < 70 else "val" if value < 85 else "test"


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.archive.is_file():
        raise SystemExit("archive missing: %s" % args.archive)
    if args.output.exists():
        if not args.force:
            raise SystemExit("output already exists; pass --force to rebuild: %s" % args.output)
        shutil.rmtree(args.output)

    stats = {"train": 0, "val": 0, "test": 0}
    fixes = []
    with zipfile.ZipFile(args.archive) as archive:
        members = archive.namelist()
        image_members = {
            Path(name).stem: name
            for name in members
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        }
        xml_members = {
            Path(name).stem: name for name in members if Path(name).suffix.lower() == ".xml"
        }
        if set(image_members) != set(xml_members):
            raise ValueError(
                "image/XML mismatch: images_only=%s xml_only=%s"
                % (sorted(set(image_members) - set(xml_members)), sorted(set(xml_members) - set(image_members)))
            )

        for stem in sorted(image_members):
            image_member = image_members[stem]
            filename = Path(image_member).name
            split = split_for(filename)
            image_output = args.output / "images" / split / filename
            label_output = args.output / "labels" / split / (stem + ".txt")
            image_output.parent.mkdir(parents=True, exist_ok=True)
            label_output.parent.mkdir(parents=True, exist_ok=True)

            with archive.open(image_member) as source, image_output.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            with Image.open(image_output) as image:
                width, height = image.size
                image.verify()

            root = ET.fromstring(archive.read(xml_members[stem]))
            labels = []
            for obj in root.findall("object"):
                class_name = (obj.findtext("name") or "").strip()
                if class_name != "gangzhu":
                    raise ValueError("unexpected class %r in %s" % (class_name, xml_members[stem]))
                box = obj.find("bndbox")
                if box is None:
                    raise ValueError("missing bndbox in %s" % xml_members[stem])
                original = [float(box.findtext(key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
                x1, y1, x2, y2 = original
                x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
                y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
                clipped = [x1, y1, x2, y2]
                if clipped != original:
                    fixes.append({"image": filename, "original": original, "clipped": clipped})
                if not (x1 < x2 and y1 < y2):
                    raise ValueError("invalid box after clipping in %s: %s" % (xml_members[stem], clipped))
                center_x = ((x1 + x2) / 2.0) / width
                center_y = ((y1 + y2) / 2.0) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                labels.append("0 %.8f %.8f %.8f %.8f" % (center_x, center_y, box_width, box_height))

            label_output.write_text("\n".join(labels) + "\n", encoding="utf-8")
            stats[split] += 1

    yaml_path = args.output / "dataset.yaml"
    hard_negative_names = []
    for source in sorted((HARD_NEGATIVES / "images").iterdir()):
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        source_label = HARD_NEGATIVES / "labels" / (source.stem + ".txt")
        if not source_label.is_file() or source_label.read_text(encoding="utf-8").strip():
            raise ValueError("hard negative must have an empty label: %s" % source)
        destination = args.output / "images" / "train" / source.name
        shutil.copy2(source, destination)
        (args.output / "labels" / "train" / (source.stem + ".txt")).write_text("", encoding="utf-8")
        hard_negative_names.append(source.name)
        stats["train"] += 1

    yaml_path.write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: steel_ball\n"
        % args.output.resolve().as_posix(),
        encoding="utf-8",
    )
    manifest = {
        "source": str(args.archive.resolve()),
        "source_sha256": archive_sha256(args.archive),
        "split_method": "sha256('real/' + filename) modulo 100; train <70, val <85, test otherwise",
        "stats": stats,
        "classes": {"0": "steel_ball"},
        "box_fixes": fixes,
        "hard_negatives": hard_negative_names,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("DATASET_70596_READY", json.dumps(manifest, ensure_ascii=True))
    print("yaml=%s" % yaml_path)


if __name__ == "__main__":
    main()
