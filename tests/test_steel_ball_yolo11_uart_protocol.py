"""Host-side checks for the K230 UART packet and result normalization logic."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "canmv" / "steel_ball_yolo11_uart.py"


def load_board_module():
    libs = types.ModuleType("libs")
    pipeline = types.ModuleType("libs.PipeLine")
    yolo = types.ModuleType("libs.YOLO")
    pipeline.PipeLine = object
    yolo.YOLO11 = object
    machine = types.ModuleType("machine")

    class FPIOA:
        UART2_TXD = 1
        UART2_RXD = 2

    class UART:
        UART2 = 2

    machine.FPIOA = FPIOA
    machine.UART = UART
    originals = {name: sys.modules.get(name) for name in ("libs", "libs.PipeLine", "libs.YOLO", "machine")}
    sys.modules.update({"libs": libs, "libs.PipeLine": pipeline, "libs.YOLO": yolo, "machine": machine})
    try:
        spec = importlib.util.spec_from_file_location("steel_ball_yolo11_uart_host_test", SCRIPT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_normalise_detections_clips_and_keeps_class_zero_only():
    app = load_board_module()
    result = (
        [[-4.2, 10.4, 40.2, 30.2], [100, 100, 20, 20]],
        [0, 1],
        [0.91, 0.88],
    )
    assert app.normalise_detections(result) == [(0, 10, 40, 30, 0.91)]


def test_uart_packet_lists_every_box_centre_in_order():
    app = load_board_module()

    class FakeUART:
        def __init__(self):
            self.messages = []

        def write(self, message):
            self.messages.append(message)

    uart = FakeUART()
    app.send_centres(uart, [(10, 20, 30, 40, 0.9), (100, 200, 21, 11, 0.8)])
    assert uart.messages == ["BALL,N=2;25,40;110,205\r\n"]


def test_tracker_requires_two_hits_then_coasts_through_short_miss():
    app = load_board_module()
    tracker = app.Tracker()
    first = tracker.update([(100, 200, 20, 20, 0.9)])
    assert first == []
    second = tracker.update([(104, 202, 20, 20, 0.8)])
    assert app.tracks_to_detections(second) == [(102, 201, 20, 20, 0.8)]
    coasted = tracker.update([])
    assert app.tracks_to_detections(coasted) == [(102, 201, 20, 20, 0.8)]


def test_tracker_drops_target_after_coast_window():
    app = load_board_module()
    tracker = app.Tracker()
    tracker.update([(10, 10, 10, 10, 0.9)])
    tracker.update([(10, 10, 10, 10, 0.9)])
    for _ in range(app.COAST_MAX):
        assert len(tracker.update([])) == 1
    assert tracker.update([]) == []
