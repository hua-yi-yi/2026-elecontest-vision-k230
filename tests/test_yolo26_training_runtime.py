import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "train_yolo26", ROOT / "training" / "scripts" / "train_yolo26.py"
)
train_yolo26 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(train_yolo26)

from ultralytics.utils.torch_utils import ModelEMA


class MixedDtypeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.register_buffer("total_ops", torch.tensor([2.0], dtype=torch.float64))


class Yolo26TrainingRuntimeTests(unittest.TestCase):
    def test_ema_updates_models_with_mixed_floating_state_dtypes(self) -> None:
        model = MixedDtypeModel()
        ema = ModelEMA(model, decay=0.5, tau=1)
        with torch.no_grad():
            model.weight.fill_(3.0)
            model.total_ops.fill_(4.0)
            ema.ema.total_ops = ema.ema.total_ops.float()

        ema.update(model)

        self.assertGreater(float(ema.ema.weight[0]), 1.0)
        self.assertEqual(float(ema.ema.total_ops[0]), 2.0)

    def test_goal_aware_stopper_stops_after_seven_stalled_goal_epochs(self) -> None:
        stopper = train_yolo26.GoalAwarePlateauStopper(
            patience=7, min_map50=0.97, min_precision=0.95, min_recall=0.90
        )
        trainer = type("Trainer", (), {})()
        trainer.stop = False
        trainer.metrics = {
            "metrics/mAP50(B)": 0.98,
            "metrics/precision(B)": 0.96,
            "metrics/recall(B)": 0.91,
        }
        for epoch in range(8):
            trainer.epoch = epoch
            trainer.fitness = 0.80
            stopper(trainer)
            if epoch < 7:
                self.assertFalse(trainer.stop)

        self.assertTrue(trainer.stop)

    def test_goal_aware_stopper_does_not_stop_below_goal(self) -> None:
        stopper = train_yolo26.GoalAwarePlateauStopper(
            patience=7, min_map50=0.97, min_precision=0.95, min_recall=0.90
        )
        trainer = type("Trainer", (), {})()
        trainer.stop = False
        trainer.metrics = {
            "metrics/mAP50(B)": 0.96,
            "metrics/precision(B)": 0.94,
            "metrics/recall(B)": 0.89,
        }
        for epoch in range(12):
            trainer.epoch = epoch
            trainer.fitness = 0.80
            stopper(trainer)

        self.assertFalse(trainer.stop)


if __name__ == "__main__":
    unittest.main()
