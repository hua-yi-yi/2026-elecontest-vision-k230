#!/usr/bin/env python3
"""Merge dataset_70596 and dataset_70816 into a leakage-resistant YOLO split."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OLD_ARCHIVE = Path("/mnt/c/Users/31806/Downloads/dataset_70596.zip")
DEFAULT_NEW_ARCHIVE = Path("/mnt/c/Users/31806/Downloads/dataset_70816.zip")
DEFAULT_OUTPUT = ROOT / "training" / "data" / "dataset_70816_merged"
HARD_NEGATIVES = ROOT / "training" / "data" / "k230_hard_examples"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = {"gangzhu", "gangqiu"}


def archive_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jpeg_size(data: bytes) -> tuple[int, int]:
    """Read JPEG dimensions without requiring Pillow during dataset preparation."""
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("not a JPEG image")
    offset = 2
    while offset < len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in (0xD8, 0xD9):
            continue
        if offset + 2 > len(data):
            break
        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            break
        if marker in range(0xC0, 0xC4):
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            if width <= 0 or height <= 0:
                raise ValueError("invalid JPEG dimensions")
            return width, height
        offset += segment_length
    raise ValueError("JPEG dimensions not found")


def split_new_stems(stems: list[str]) -> dict[str, str]:
    """Reserve the latest 15% for test and preceding 15% for validation."""
    train_end = round(len(stems) * 0.70)
    val_end = round(len(stems) * 0.85)
    return {
        stem: "train" if index < train_end else "val" if index < val_end else "test"
        for index, stem in enumerate(stems)
    }


def convert_archive(
    archive_path: Path,
    output: Path,
    source_name: str,
    split_by_stem: dict[str, str],
    stats: dict[str, int],
    fixes: list[dict],
) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.namelist()
        image_members = {
            Path(name).stem: name
            for name in members
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        }
        xml_members = {
            Path(name).stem: name
            for name in members
            if Path(name).suffix.lower() == ".xml"
        }
        if set(image_members) != set(xml_members):
            raise ValueError(
                "%s image/XML mismatch: images_only=%s xml_only=%s"
                % (
                    source_name,
                    sorted(set(image_members) - set(xml_members)),
                    sorted(set(xml_members) - set(image_members)),
                )
            )
        if set(image_members) != set(split_by_stem):
            raise ValueError("%s split does not cover every image" % source_name)

        object_count = 0
        image_hashes = {}
        for stem in sorted(image_members):
            image_member = image_members[stem]
            image_data = archive.read(image_member)
            suffix = Path(image_member).suffix.lower()
            if suffix not in {".jpg", ".jpeg"}:
                raise ValueError("unsupported non-JPEG image: %s" % image_member)
            width, height = jpeg_size(image_data)
            split = split_by_stem[stem]
            output_stem = "%s_%s" % (source_name, stem)
            image_output = output / "images" / split / (output_stem + suffix)
            label_output = output / "labels" / split / (output_stem + ".txt")
            image_output.parent.mkdir(parents=True, exist_ok=True)
            label_output.parent.mkdir(parents=True, exist_ok=True)
            image_output.write_bytes(image_data)

            root = ET.fromstring(archive.read(xml_members[stem]))
            labels = []
            for obj in root.findall("object"):
                class_name = (obj.findtext("name") or "").strip()
                if class_name not in CLASS_NAMES:
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
                    fixes.append(
                        {
                            "source": source_name,
                            "image": Path(image_member).name,
                            "original": original,
                            "clipped": clipped,
                        }
                    )
                if not (x1 < x2 and y1 < y2):
                    raise ValueError("invalid box after clipping in %s: %s" % (xml_members[stem], clipped))
                labels.append(
                    "0 %.8f %.8f %.8f %.8f"
                    % (
                        ((x1 + x2) / 2.0) / width,
                        ((y1 + y2) / 2.0) / height,
                        (x2 - x1) / width,
                        (y2 - y1) / height,
                    )
                )
            if not labels:
                raise ValueError("positive image has no objects: %s" % image_member)
            label_output.write_text("\n".join(labels) + "\n", encoding="utf-8")
            object_count += len(labels)
            stats[split] += 1
            image_hashes[output_stem] = hashlib.sha256(image_data).hexdigest()

    return {
        "archive": str(archive_path.resolve()),
        "sha256": archive_sha256(archive_path),
        "images": len(image_hashes),
        "objects": object_count,
        "image_hashes": image_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-archive", type=Path, default=DEFAULT_OLD_ARCHIVE)
    parser.add_argument("--new-archive", type=Path, default=DEFAULT_NEW_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for archive in (args.old_archive, args.new_archive):
        if not archive.is_file():
            raise SystemExit("archive missing: %s" % archive)
    if args.output.exists():
        if not args.force:
            raise SystemExit("output already exists; pass --force to rebuild: %s" % args.output)
        shutil.rmtree(args.output)

    with zipfile.ZipFile(args.old_archive) as archive:
        old_stems = sorted(
            Path(name).stem for name in archive.namelist()
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        )
    with zipfile.ZipFile(args.new_archive) as archive:
        new_stems = sorted(
            Path(name).stem for name in archive.namelist()
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        )

    stats = {"train": 0, "val": 0, "test": 0}
    fixes = []
    sources = {
        "dataset_70596": convert_archive(
            args.old_archive,
            args.output,
            "dataset_70596",
            {stem: "train" for stem in old_stems},
            stats,
            fixes,
        ),
        "dataset_70816": convert_archive(
            args.new_archive,
            args.output,
            "dataset_70816",
            split_new_stems(new_stems),
            stats,
            fixes,
        ),
    }

    all_hashes = [digest for source in sources.values() for digest in source["image_hashes"].values()]
    if len(all_hashes) != len(set(all_hashes)):
        raise ValueError("duplicate image content found across merged sources")

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

    yaml_path = args.output / "dataset.yaml"
    yaml_path.write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: steel_ball\n"
        % args.output.resolve().as_posix(),
        encoding="utf-8",
    )
    for source in sources.values():
        source.pop("image_hashes")
    manifest = {
        "sources": sources,
        "split_method": (
            "all dataset_70596 positives train; dataset_70816 sorted capture order "
            "70% train, 15% val, 15% test; test excluded from training and PTQ"
        ),
        "stats": stats,
        "classes": {"0": "steel_ball"},
        "source_classes": sorted(CLASS_NAMES),
        "box_fixes": fixes,
        "hard_negatives": hard_negative_names,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("DATASET_70816_MERGED_READY", json.dumps(manifest, ensure_ascii=True))
    print("yaml=%s" % yaml_path)


if __name__ == "__main__":
    main()
