#!/usr/bin/env python3
"""Compare YOLO26 confidence counts across exported input resolutions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="SIZE=path.onnx")
    parser.add_argument("--image", action="append", type=Path, required=True)
    args = parser.parse_args()

    for spec in args.model:
        size_text, model_text = spec.split("=", 1)
        size = int(size_text)
        session = ort.InferenceSession(model_text)
        input_name = session.get_inputs()[0].name
        print("MODEL", size)
        for path in args.image:
            image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
            tensor = np.transpose(np.asarray(image, dtype=np.float32) / 255.0, (2, 0, 1))[None]
            output = session.run(None, {input_name: tensor})[0]
            scores = output[0, :, 4]
            print(
                path.name,
                "max", round(float(scores.max()), 4),
                "n010", int((scores >= 0.10).sum()),
                "n015", int((scores >= 0.15).sum()),
                "n020", int((scores >= 0.20).sum()),
                "n030", int((scores >= 0.30).sum()),
            )


if __name__ == "__main__":
    main()
