#!/usr/bin/env python3
"""Export a YOLO11 checkpoint with the legacy ONNX tracer for nncase 2.11.

Recent PyTorch versions route Ultralytics through the dynamo ONNX exporter,
which may emit a Reshape ``allowzero`` attribute unsupported by nncase 2.11.
This uses the legacy tracer explicitly and validates an opset-13 graph.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "training" / "runs" / "detect" / "steel_ball_reference_yolo11n_1024" / "weights" / "best.pt"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "export" / "steel_ball_reference_yolo11n_1024_legacy.onnx"


def shape(value_info):
    return [item.dim_value if item.dim_value else item.dim_param for item in value_info.type.tensor_type.shape.dim]


class DetectionOutput(torch.nn.Module):
    """Export only YOLO's decoded detection tensor, not its training feature maps."""

    def __init__(self, model, normalise_raw_uint8=False):
        super().__init__()
        self.model = model
        self.normalise_raw_uint8 = normalise_raw_uint8

    def forward(self, image):
        # CanMV's YOLO wrapper supplies raw RGBP uint8 pixels.  Keeping this
        # divide inside the graph avoids relying on nncase's optional external
        # preprocessor, which produced raw 0..255 input on the board.
        if self.normalise_raw_uint8:
            image = image / 255.0
        result = self.model(image)
        if isinstance(result, (tuple, list)):
            return result[0]
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a fixed ONNX graph compatible with nncase 2.11.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument(
        "--raw-uint8-input", action="store_true",
        help="embed /255 normalisation so the exported model accepts raw camera pixels",
    )
    args = parser.parse_args()
    if not args.weights.is_file():
        raise SystemExit("weights missing: %s" % args.weights)

    wrapper = YOLO(str(args.weights))
    model = DetectionOutput(wrapper.model.fuse().eval().cpu(), args.raw_uint8_input)
    sample = torch.zeros((1, 3, args.imgsz, args.imgsz), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            sample,
            str(args.output),
            input_names=["images"],
            output_names=["output0"],
            opset_version=13,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
    graph = onnx.load(str(args.output))
    onnx.checker.check_model(graph)
    opsets = [{"domain": item.domain, "version": item.version} for item in graph.opset_import]
    if any(item["domain"] == "" and item["version"] != 13 for item in opsets):
        raise RuntimeError("unexpected default ONNX opset: %s" % opsets)
    if len(graph.graph.output) != 1:
        raise RuntimeError("expected one decoded detection output, got %d" % len(graph.graph.output))
    report = {
        "weights": str(args.weights.resolve()),
        "onnx": str(args.output.resolve()),
        "input_size": [args.imgsz, args.imgsz],
        "raw_uint8_input": args.raw_uint8_input,
        "opsets": opsets,
        "inputs": [{"name": item.name, "shape": shape(item)} for item in graph.graph.input],
        "outputs": [{"name": item.name, "shape": shape(item)} for item in graph.graph.output],
    }
    args.output.with_suffix(".onnx.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
