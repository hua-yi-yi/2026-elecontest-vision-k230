#!/usr/bin/env python3
"""Compile fixed-shape YOLO ONNX -> K230 kmodel via nncase (optional host step)."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PTQ_TYPES = {
    "0": ("uint8", "uint8"),
    "1": ("uint8", "int16"),
    "2": ("int16", "uint8"),
}


def generate_calibration_data(dataset: Path, shape: list[int], count: int, dtype=np.uint8):
    paths = sorted(
        p for p in Path(dataset).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(paths) < count:
        raise ValueError(f"need {count} calibration images, found {len(paths)} in {dataset}")
    samples = []
    for path in paths[:count]:
        image = Image.open(path).convert("RGB").resize((shape[3], shape[2]), Image.Resampling.BILINEAR)
        tensor = np.transpose(np.asarray(image, dtype=dtype), (2, 0, 1))[None]
        samples.append([tensor])
    return samples


def simplify_onnx(model_path: Path, output_path: Path, input_shape: list[int]) -> None:
    import onnx
    import onnxsim

    model = onnx.load(str(model_path))
    initializer_names = {node.name for node in model.graph.initializer}
    inputs = [node for node in model.graph.input if node.name not in initializer_names]
    if len(inputs) != 1:
        raise ValueError(f"expected one model input, found {len(inputs)}")
    model = onnx.shape_inference.infer_shapes(model)
    simplified, valid = onnxsim.simplify(model, input_shapes={inputs[0].name: input_shape})
    if not valid:
        raise RuntimeError("onnxsim validation failed")
    onnx.save_model(simplified, str(output_path))


def compile_model(args: argparse.Namespace) -> dict:
    import nncase

    if args.no_external_preprocess:
        raise ValueError(
            "nncase 2.11 applies input_type and input_shape only when preprocess=True. "
            "CanMV YOLO11 supplies uint8 frames, so --no-external-preprocess emits an "
            "incompatible float-input KModel. Keep external preprocessing enabled; for an "
            "ONNX graph that already divides by 255, pass --input-std 1 instead."
        )

    width = ((args.input_width + 31) // 32) * 32
    height = ((args.input_height + 31) // 32) * 32
    input_shape = [1, 3, height, width]
    with tempfile.TemporaryDirectory(prefix="k230-steel-") as temp_dir:
        simplified_path = Path(temp_dir) / "simplified.onnx"
        simplify_onnx(args.model, simplified_path, input_shape)

        options = nncase.CompileOptions()
        options.target = args.target
        options.preprocess = True
        options.swapRB = False
        options.input_shape = input_shape
        options.input_type = "uint8"
        options.input_range = [0, 255]
        options.mean = [0, 0, 0]
        options.std = [args.input_std, args.input_std, args.input_std]
        options.input_layout = "NCHW"

        compiler = nncase.Compiler(options)
        compiler.import_onnx(simplified_path.read_bytes(), nncase.ImportOptions())
        quant_types = PTQ_TYPES.get(args.ptq_option)
        if quant_types:
            samples = generate_calibration_data(args.dataset, input_shape, args.samples, np.uint8)
            ptq = nncase.PTQTensorOptions()
            ptq.samples_count = len(samples)
            ptq.calibrate_method = "NoClip"
            ptq.quant_type, ptq.w_quant_type = quant_types
            # Match the proven target_best.kmodel pipeline. Per-channel weight
            # ranges preserve small-object detector heads far better than one
            # shared range for all channels.
            ptq.export_weight_range_by_channel = True
            ptq.set_tensor_data(samples)
            compiler.use_ptq(ptq)
        compiler.compile()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(compiler.gencode_tobytes())

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    report = {
        "model": str(args.model.resolve()),
        "output": str(args.output.resolve()),
        "target": args.target,
        "ptq_option": args.ptq_option,
        "activation_quant_type": quant_types[0] if quant_types else None,
        "weight_quant_type": quant_types[1] if quant_types else None,
        "calibration_samples": args.samples if quant_types else 0,
        "input_shape": input_shape,
        "compile_settings": {
            "input_type": "uint8",
            "input_range": [0, 255],
            "mean": [0, 0, 0],
            "std": [args.input_std, args.input_std, args.input_std],
            "input_layout": "NCHW",
            "swapRB": False,
        },
        "bytes": args.output.stat().st_size,
        "sha256": digest,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="calibration images root")
    parser.add_argument("--input-width", type=int, default=320)
    parser.add_argument("--input-height", type=int, default=320)
    parser.add_argument("--target", default="k230")
    parser.add_argument(
        "--ptq-option", choices=("none", "0", "1", "2"), default="0",
        help="none=FP32, 0=U8W8, 1=U8W16, 2=I16W8",
    )
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument(
        "--no-external-preprocess", action="store_true",
        help="unsupported with nncase 2.11; retained only to fail clearly instead of emitting an invalid KModel",
    )
    parser.add_argument(
        "--input-std", type=float, default=255.0,
        help="external preprocessing standard deviation; use 1 for a raw-input ONNX graph",
    )
    args = parser.parse_args()
    try:
        import nncase  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "nncase not installed on this host. Export ONNX first, then convert on a machine "
            "with nncase (or LCKFB/CanMV convert tools). See README."
        ) from exc
    report = compile_model(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
