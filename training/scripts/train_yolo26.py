#!/usr/bin/env python3
"""Train the same NMS-free YOLO26 family used by the reference project."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils import torch_utils


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "training" / "data" / "reference_v1" / "steel_ball_reference.yaml"
DEFAULT_WEIGHTS = ROOT / "yolo26n.pt"
DEFAULT_PROJECT = ROOT / "training" / "runs" / "detect"


def unwrap_training_model(model):
    """Return the underlying model across plain, compiled, and parallel wrappers."""
    while True:
        if hasattr(model, "_orig_mod") and isinstance(model._orig_mod, torch.nn.Module):
            model = model._orig_mod
        elif hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
            model = model.module
        else:
            return model


def install_mixed_dtype_ema_patch() -> None:
    """Group EMA updates by dtype to support THOP's float64 profiling buffers."""
    model_ema = torch_utils.ModelEMA
    if getattr(model_ema.update, "_steel_ball_mixed_dtype_safe", False):
        return

    def update(self, model) -> None:
        if not self.enabled:
            return
        self.updates += 1
        decay = self.decay(self.updates)
        model_state = unwrap_training_model(model).state_dict()
        groups = {}
        for key, ema_value in self.ema.state_dict().items():
            if not ema_value.dtype.is_floating_point:
                continue
            if key.rsplit(".", 1)[-1] in {"total_ops", "total_params"}:
                continue
            model_value = model_state[key]
            if ema_value.dtype != model_value.dtype:
                raise RuntimeError(
                    "EMA dtype mismatch for %s: %s != %s"
                    % (key, ema_value.dtype, model_value.dtype)
                )
            group = groups.setdefault((ema_value.device, ema_value.dtype), ([], []))
            group[0].append(ema_value)
            group[1].append(model_value)

        for (device, _dtype), (ema_values, model_values) in groups.items():
            use_foreach = torch_utils.TORCH_2_0 and (torch_utils.TORCH_2_4 or device.type != "mps")
            if use_foreach:
                torch._foreach_lerp_(ema_values, model_values, 1 - decay)
            else:
                for ema_value, model_value in zip(ema_values, model_values):
                    ema_value.mul_(decay).add_(model_value, alpha=1 - decay)

    update._steel_ball_mixed_dtype_safe = True
    model_ema.update = update


install_mixed_dtype_ema_patch()


class GoalAwarePlateauStopper:
    """Stop only after target metrics are reached and fitness then plateaus."""

    def __init__(
        self,
        patience: int = 7,
        min_map50: float = 0.97,
        min_precision: float = 0.95,
        min_recall: float = 0.90,
        min_delta: float = 1e-4,
    ) -> None:
        self.patience = patience
        self.min_map50 = min_map50
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.min_delta = min_delta
        self.best_fitness = float("-inf")
        self.goal_reached = False
        self.stalled_epochs = 0

    def __call__(self, trainer) -> None:
        metrics = trainer.metrics
        fitness = float(trainer.fitness)
        improved = fitness > self.best_fitness + self.min_delta
        if improved:
            self.best_fitness = fitness
            self.stalled_epochs = 0

        meets_goal = (
            float(metrics.get("metrics/mAP50(B)", 0.0)) >= self.min_map50
            and float(metrics.get("metrics/precision(B)", 0.0)) >= self.min_precision
            and float(metrics.get("metrics/recall(B)", 0.0)) >= self.min_recall
        )
        if meets_goal and not self.goal_reached:
            self.goal_reached = True
            self.stalled_epochs = 0
        elif self.goal_reached and not improved:
            self.stalled_epochs += 1

        print(
            "GOAL_EARLY_STOP epoch=%d goal=%s stalled=%d/%d best_fitness=%.6f"
            % (
                trainer.epoch + 1,
                "yes" if self.goal_reached else "no",
                self.stalled_epochs,
                self.patience,
                self.best_fitness,
            ),
            flush=True,
        )
        if self.goal_reached and self.stalled_epochs >= self.patience:
            trainer.stop = True
            print("GOAL_EARLY_STOP triggered=plateau_after_goal", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO26n steel-ball detector.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="steel_ball_reference_yolo26n_1024")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--goal-patience", type=int, default=7)
    parser.add_argument("--goal-map50", type=float, default=0.97)
    parser.add_argument("--goal-precision", type=float, default=0.95)
    parser.add_argument("--goal-recall", type=float, default=0.90)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--scale", type=float, default=0.25)
    parser.add_argument("--mosaic", type=float, default=0.30)
    parser.add_argument("--close-mosaic", type=int, default=10)
    args = parser.parse_args()
    if not args.data.is_file() or not args.weights.is_file() or (args.resume and not args.resume.is_file()):
        raise SystemExit("dataset or YOLO26 pretrained weights missing")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    print("data=%s weights=%s device=%s" % (args.data, args.weights, device))
    model = YOLO(str(args.resume or args.weights))
    model.add_callback(
        "on_fit_epoch_end",
        GoalAwarePlateauStopper(
            patience=args.goal_patience,
            min_map50=args.goal_map50,
            min_precision=args.goal_precision,
            min_recall=args.goal_recall,
        ),
    )
    if args.resume:
        model.train(resume=str(args.resume), workers=0, device=device, patience=0)
        return
    model.train(
        data=str(args.data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=device, workers=0, project=str(DEFAULT_PROJECT), name=args.name,
        exist_ok=True, pretrained=True, optimizer="AdamW", lr0=args.lr0, lrf=0.01,
        patience=0, seed=20260727, deterministic=True,
        degrees=5.0, translate=0.06, scale=args.scale, shear=0.5, perspective=0.0,
        hsv_h=0.006, hsv_s=0.18, hsv_v=0.20, fliplr=0.5, flipud=0.0,
        mosaic=args.mosaic, close_mosaic=args.close_mosaic, mixup=0.0, copy_paste=0.0,
        erasing=0.02, cache="ram", amp=True, plots=True,
    )


if __name__ == "__main__":
    main()
