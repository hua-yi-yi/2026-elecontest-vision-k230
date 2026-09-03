from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "canmv" / "steel_ball_yolo26_uart_epoch19.py"


def load_canmv_script():
    libs = types.ModuleType("libs")
    pipeline_module = types.ModuleType("libs.PipeLine")
    pipeline_module.PipeLine = object
    ai_base_module = types.ModuleType("libs.AIBase")
    ai_base_module.AIBase = object
    ai2d_module = types.ModuleType("libs.AI2D")
    ai2d_module.Ai2d = object
    machine = types.ModuleType("machine")
    machine.FPIOA = object
    machine.UART = types.SimpleNamespace(UART1=1, UART2=2)
    nncase_runtime = types.ModuleType("nncase_runtime")
    ulab = types.ModuleType("ulab")
    ulab_numpy = types.ModuleType("ulab.numpy")

    stubs = {
        "libs": libs,
        "libs.PipeLine": pipeline_module,
        "libs.AIBase": ai_base_module,
        "libs.AI2D": ai2d_module,
        "machine": machine,
        "nncase_runtime": nncase_runtime,
        "ulab": ulab,
        "ulab.numpy": ulab_numpy,
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("steel_ball_canmv_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_geometry_rejects_large_and_wide_false_boxes():
    module = load_canmv_script()
    predicate = getattr(module, "detection_passes_geometry", lambda detection: True)

    assert predicate((120, 80, 24, 22, 0.75))
    assert not predicate((20, 160, 190, 92, 0.80))
    assert not predicate((300, 250, 70, 12, 0.70))


def test_tracker_requires_three_consistent_hits_for_medium_score():
    module = load_canmv_script()
    tracker = module.Tracker()
    detection = (120, 80, 24, 22, 0.55)

    assert tracker.update([detection]) == []
    assert tracker.update([detection]) == []
    stable = tracker.update([detection])

    assert len(stable) == 1


def test_tracker_smooths_displayed_confidence():
    module = load_canmv_script()
    tracker = module.Tracker()

    tracker.update([(120, 80, 24, 22, 0.80)])
    tracker.update([(121, 81, 24, 22, 0.50)])
    stable = tracker.update([(120, 80, 24, 22, 0.70)])

    assert len(stable) == 1
    assert 0.60 < stable[0][4] < 0.75


def test_kalman_prediction_matches_fast_motion():
    module = load_canmv_script()
    tracker = module.Tracker()
    tracker.update([(40, 100, 20, 20, 0.90)])
    tracker.update([(70, 100, 20, 20, 0.90)])
    stable = tracker.update([(120, 100, 20, 20, 0.90)])

    assert len(stable) == 1
    assert stable[0][0] >= 105


def test_kalman_coasts_confirmed_track_for_configured_limit():
    module = load_canmv_script()
    tracker = module.Tracker()
    detection = (100, 100, 20, 20, 0.90)
    tracker.update([detection])
    tracker.update([detection])
    tracker.update([detection])

    predictions = []
    for _ in range(module.COAST_MAX):
        stable = tracker.update([])
        assert len(stable) == 1
        predictions.append(stable[0])

    assert predictions[-1][4] < detection[4]
    assert tracker.update([]) == []


def test_kalman_prediction_stays_inside_frame():
    module = load_canmv_script()
    tracker = module.Tracker()
    tracker.update([(490, 270, 20, 18, 0.90)])
    tracker.update([(510, 280, 2, 2, 0.90)])
    tracker.update([(510, 280, 2, 2, 0.90)])
    stable = tracker.update([])

    assert len(stable) == 1
    x, y, width, height, _ = stable[0]
    assert 0 <= x < module.AI_CAPTURE_SIZE[0]
    assert 0 <= y < module.AI_CAPTURE_SIZE[1]
    assert x + width <= module.AI_CAPTURE_SIZE[0]
    assert y + height <= module.AI_CAPTURE_SIZE[1]


class FakeUART:
    def __init__(self):
        self.messages = []

    def write(self, message):
        self.messages.append(bytes(message))


def test_uart_sends_highest_confidence_centre_as_big_endian_frame():
    module = load_canmv_script()
    uart = FakeUART()

    module.send_centres(uart, [
        (100, 50, 20, 10, 0.60),
        (240, 120, 32, 24, 0.90),
    ])

    assert uart.messages == [bytes((0xAA, 0x55, 0x00, 0xA0, 0x00, 0x6E, 0x0D, 0x0A))]


def test_uart_does_not_send_when_there_is_no_detection():
    module = load_canmv_script()
    uart = FakeUART()

    module.send_centres(uart, [])

    assert uart.messages == []


def test_uart_clamps_scaled_centre_to_mspm0_coordinate_bounds():
    module = load_canmv_script()
    uart = FakeUART()

    module.send_centres(uart, [(511, 287, 20, 20, 0.90)])

    assert uart.messages == [bytes((0xAA, 0x55, 0x01, 0x3F, 0x00, 0xEF, 0x0D, 0x0A))]
