import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "training_dashboard", ROOT / "training" / "scripts" / "training_dashboard.py"
)
dashboard = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(dashboard)


class TrainingDashboardTests(unittest.TestCase):
    def test_parse_gpu_output_returns_real_percentages(self) -> None:
        parser = getattr(dashboard, "parse_gpu_output", None)
        self.assertIsNotNone(parser)
        gpu = parser("0, NVIDIA GeForce RTX 5060, 18, 4242, 8151, 40, 37\n")

        self.assertTrue(gpu["available"])
        self.assertEqual(gpu["name"], "NVIDIA GeForce RTX 5060")
        self.assertEqual(gpu["utilization"], 18)
        self.assertAlmostEqual(gpu["memory_percent"], 4242 / 8151 * 100)
        self.assertEqual(gpu["fan_percent"], 37)

    def test_parse_progress_extracts_live_training_fields(self) -> None:
        text = (
            "\x1b[K       7/60       1.9G     0.4336     0.2327  "
            "0.0006238         39       1024: 49% ━━━━━ 346/703 2.4it/s 2:06<2:27"
        )

        parser = getattr(dashboard, "parse_progress", None)
        self.assertIsNotNone(parser)
        progress = parser(text)

        self.assertEqual(progress["epoch_current"], 7)
        self.assertEqual(progress["epoch_total"], 60)
        self.assertEqual(progress["batch_current"], 346)
        self.assertEqual(progress["batch_total"], 703)
        self.assertEqual(progress["gpu_memory"], "1.9G")
        self.assertAlmostEqual(progress["box_loss"], 0.4336)
        self.assertAlmostEqual(progress["cls_loss"], 0.2327)
        self.assertAlmostEqual(progress["batch_rate"], 2.4)
        self.assertEqual(progress["eta"], "2:27")

    def test_status_is_running_only_while_log_is_fresh(self) -> None:
        classifier = getattr(dashboard, "classify_status", None)
        self.assertIsNotNone(classifier)
        self.assertEqual(
            classifier("training output", age_seconds=2, has_results=True),
            "running",
        )
        self.assertEqual(
            classifier("training output", age_seconds=40, has_results=True),
            "completed",
        )

    def test_status_preserves_completed_and_failed_runs_in_history(self) -> None:
        classifier = getattr(dashboard, "classify_status", None)
        self.assertIsNotNone(classifier)
        self.assertEqual(
            classifier("GOAL_EARLY_STOP triggered=plateau_after_goal", 1, True),
            "completed",
        )
        self.assertEqual(
            classifier("Traceback\nRuntimeError: broken", 40, True),
            "failed",
        )


if __name__ == "__main__":
    unittest.main()
