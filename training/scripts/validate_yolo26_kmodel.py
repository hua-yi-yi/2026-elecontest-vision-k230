#!/usr/bin/env python3
"""Compare a YOLO26 end-to-end KModel with its ONNX source on real images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def prepare_image(path: Path, width: int, height: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    return np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1))[None]


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def decode(output: np.ndarray, confidence: float) -> list[dict]:
    if output.shape != (1, 300, 6):
        raise ValueError(f"expected [1,300,6], got {output.shape}")
    detections = []
    for row in output[0]:
        score = float(row[4])
        if score < confidence:
            continue
        detections.append(
            {
                "box": row[:4].astype(np.float64),
                "score": score,
                "class_id": int(round(float(row[5]))),
            }
        )
    return sorted(detections, key=lambda item: item["score"], reverse=True)


def compare(reference: list[dict], candidate: list[dict], minimum_iou: float) -> dict:
    remaining = list(candidate)
    matches = []
    unmatched_reference = []
    for item in reference:
        choices = [
            (index, box_iou(item["box"], other["box"]))
            for index, other in enumerate(remaining)
            if other["class_id"] == item["class_id"]
        ]
        if not choices:
            unmatched_reference.append(item)
            continue
        index, iou = max(choices, key=lambda choice: choice[1])
        if iou < minimum_iou:
            unmatched_reference.append(item)
            continue
        other = remaining.pop(index)
        matches.append(
            {
                "iou": iou,
                "score_error": abs(item["score"] - other["score"]),
                "max_coordinate_error": float(np.abs(item["box"] - other["box"]).max()),
            }
        )
    return {
        "passed": not unmatched_reference and not remaining,
        "matches": matches,
        "unmatched_reference": len(unmatched_reference),
        "unmatched_candidate": len(remaining),
    }


def ground_truth_boxes(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, center_x, center_y, box_width, box_height = map(float, line.split())
        boxes.append(
            np.array(
                [
                    (center_x - box_width / 2) * width,
                    (center_y - box_height / 2) * height,
                    (center_x + box_width / 2) * width,
                    (center_y + box_height / 2) * height,
                    class_id,
                ],
                dtype=np.float64,
            )
        )
    return boxes


def detection_metrics(detections: list[dict], targets: list[np.ndarray], minimum_iou: float) -> dict:
    remaining = list(detections)
    true_positives = 0
    for target in targets:
        choices = [
            (index, box_iou(target, item["box"]))
            for index, item in enumerate(remaining)
            if item["class_id"] == int(target[4])
        ]
        if not choices:
            continue
        index, iou = max(choices, key=lambda choice: choice[1])
        if iou >= minimum_iou:
            remaining.pop(index)
            true_positives += 1
    false_positives = len(remaining)
    false_negatives = len(targets) - true_positives
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--kmodel", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--minimum-iou", type=float, default=0.98)
    parser.add_argument("--ground-truth-iou", type=float, default=0.50)
    parser.add_argument("--input-width", type=int, default=416)
    parser.add_argument("--input-height", type=int, default=416)
    args = parser.parse_args()

    import nncase
    import onnxruntime as ort

    image_paths = sorted(
        path for path in args.images.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )[: args.limit]
    if not image_paths:
        raise SystemExit(f"no images found in {args.images}")

    session = ort.InferenceSession(str(args.onnx))
    input_name = session.get_inputs()[0].name
    simulator = nncase.Simulator()
    simulator.load_model(args.kmodel.read_bytes())
    expected_input_shape = [1, 3, args.input_height, args.input_width]
    if list(simulator.get_input_shape(0)) != expected_input_shape:
        raise RuntimeError(f"unexpected input shape: {simulator.get_input_shape(0)}")
    if list(simulator.get_output_shape(0)) != [1, 300, 6]:
        raise RuntimeError(f"unexpected output shape: {simulator.get_output_shape(0)}")

    results = []
    totals = {
        "onnx": {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
        "kmodel": {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
    }
    for path in image_paths:
        uint8_input = prepare_image(path, args.input_width, args.input_height)
        onnx_output = session.run(None, {input_name: uint8_input.astype(np.float32) / 255.0})[0]
        simulator.set_input_tensor(0, nncase.RuntimeTensor.from_numpy(uint8_input))
        simulator.run()
        kmodel_output = simulator.get_output_tensor(0).to_numpy()
        if not np.isfinite(kmodel_output).all():
            raise RuntimeError(f"non-finite KModel output for {path.name}")
        onnx_detections = decode(onnx_output, args.confidence)
        kmodel_detections = decode(kmodel_output, args.confidence)
        comparison = compare(onnx_detections, kmodel_detections, args.minimum_iou)
        target_metrics = None
        if args.labels:
            targets = ground_truth_boxes(
                args.labels / (path.stem + ".txt"), args.input_width, args.input_height
            )
            target_metrics = {
                "onnx": detection_metrics(onnx_detections, targets, args.ground_truth_iou),
                "kmodel": detection_metrics(kmodel_detections, targets, args.ground_truth_iou),
            }
            for runtime in totals:
                for metric, value in target_metrics[runtime].items():
                    totals[runtime][metric] += value
        results.append(
            {
                "image": path.name,
                "onnx_detections": len(onnx_detections),
                "kmodel_detections": len(kmodel_detections),
                "ground_truth": target_metrics,
                **comparison,
            }
        )

    if args.labels:
        passed = (
            totals["kmodel"]["false_negatives"] <= totals["onnx"]["false_negatives"]
            and totals["kmodel"]["false_positives"] <= totals["onnx"]["false_positives"]
        )
    else:
        passed = all(item["passed"] for item in results)
    report = {
        "onnx": args.onnx.as_posix(),
        "kmodel": args.kmodel.as_posix(),
        "input_shape": expected_input_shape,
        "output_shape": [1, 300, 6],
        "confidence": args.confidence,
        "minimum_iou": args.minimum_iou,
        "ground_truth_iou": args.ground_truth_iou if args.labels else None,
        "passed": passed,
        "tensor_comparison_passed": all(item["passed"] for item in results),
        "ground_truth_totals": totals if args.labels else None,
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("YOLO26 KModel validation failed")


if __name__ == "__main__":
    main()
