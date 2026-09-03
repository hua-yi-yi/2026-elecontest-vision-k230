"""K230 CanMV v1.6 steel-ball detector for the YOLO26 epoch-19 KModel.

TF-card layout:
  /sdcard/models/steel_ball_yolo26n_epoch19_416_i16w8.kmodel
  /sdcard/steel_ball_yolo26_uart_epoch19.py

The model has an end-to-end output shaped [1, 300, 6]. Each row is
[x1, y1, x2, y2, confidence, class_id]. It must not be decoded with the
YOLOv8/YOLO11 [1, 5, N] wrappers.
"""

from libs.PipeLine import PipeLine
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from machine import FPIOA, UART
import gc
import os
import sys
import nncase_runtime as nn
import ulab.numpy as np


SCRIPT_VERSION = "STEEL-BALL-YOLO26-EPOCH19-416-I16W8-V4"
KMODEL_PATH = "/sdcard/models/steel_ball_yolo26n_epoch19_416_i16w8.kmodel"
MODEL_INPUT_SIZE = [416, 416]
AI_CAPTURE_SIZE = [512, 288]
DISPLAY_MODE = "virt"
DISPLAY_SIZE = [800, 480]
CONFIDENCE_THRESHOLD = 0.15
DISPLAY_CONFIDENCE_THRESHOLD = 0.30
FAST_CONFIRM_THRESHOLD = 0.70
MAX_BOXES = 100
ENABLE_TRACKING = True

CONFIRM_HITS = 3
COAST_MAX = 2
MATCH_DISTANCE = 36
EMA_ALPHA = 0.50
SCORE_EMA_ALPHA = 0.35
ENABLE_KALMAN = True
KALMAN_PROCESS_POSITION = 4.0
KALMAN_PROCESS_VELOCITY = 9.0
KALMAN_MEASUREMENT_NOISE = 16.0
KALMAN_MAX_EXTRA_DISTANCE = 48
KALMAN_CONFIDENCE_DECAY = 0.85

MIN_BOX_SIDE = 5
MIN_ASPECT_RATIO = 0.60
MAX_ASPECT_RATIO = 1.65
MAX_BOX_WIDTH_RATIO = 0.22
MAX_BOX_HEIGHT_RATIO = 0.35
MAX_BOX_AREA_RATIO = 0.08

ENABLE_UART = True
UART_ID = UART.UART1
UART_BAUD = 115200
UART1_TX_GPIO = 9
UART1_RX_GPIO = 10
UART_SEND_EVERY_N_FRAMES = 1
def file_exists(path):
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def model_input(frame):
    if hasattr(frame, "to_numpy_ref"):
        return frame.to_numpy_ref()
    return frame


def init_uart():
    if not ENABLE_UART:
        return None
    fpioa = FPIOA()
    fpioa.set_function(UART1_TX_GPIO, FPIOA.UART1_TXD)
    fpioa.set_function(UART1_RX_GPIO, FPIOA.UART1_RXD)
    return UART(UART_ID, baudrate=UART_BAUD, bits=UART.EIGHTBITS,
                parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE, timeout=100)


class Yolo26Detector(AIBase):
    """Decode the YOLO26 end-to-end tensor without applying NMS again."""

    def __init__(self, kmodel_path, model_input_size, rgb888p_size, debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        self.model_input_size = model_input_size
        self.rgb888p_size = rgb888p_size
        self.output_logged = False
        self.last_max_score = 0.0
        self.last_count_005 = 0
        self.last_count_010 = 0
        self.last_count_020 = 0
        self.last_count_030 = 0
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(
            nn.ai2d_format.NCHW_FMT,
            nn.ai2d_format.NCHW_FMT,
            np.uint8,
            np.uint8,
        )

    def config_preprocess(self, input_image_size=None):
        source_size = input_image_size if input_image_size else self.rgb888p_size
        self.ai2d.resize(
            interp_method=nn.interp_method.tf_bilinear,
            interp_mode=nn.interp_mode.half_pixel,
        )
        self.ai2d.build(
            [1, 3, source_size[1], source_size[0]],
            [1, 3, self.model_input_size[1], self.model_input_size[0]],
        )

    def postprocess(self, results):
        if not results or len(results) != 1:
            raise RuntimeError("expected one YOLO26 output tensor")
        output = results[0]
        shape = getattr(output, "shape", None)
        if not self.output_logged:
            print("stage=KPU_OUTPUT shape=%s dtype=%s" % (
                shape, getattr(output, "dtype", None),
            ))
            self.output_logged = True
        if shape is None or len(shape) != 3 or shape[0] != 1 or shape[2] != 6:
            raise RuntimeError("expected KPU output [1,300,6], got %s" % (shape,))

        rows = output[0]
        source_w, source_h = self.rgb888p_size
        model_w, model_h = self.model_input_size
        scale_x = source_w / model_w
        scale_y = source_h / model_h
        detections = []
        max_score = 0.0
        count_005 = 0
        count_010 = 0
        count_020 = 0
        count_030 = 0
        for index in range(shape[1]):
            row = rows[index]
            score = float(row[4])
            class_id = int(round(float(row[5])))
            if class_id == 0:
                max_score = max(max_score, score)
                if score >= 0.05:
                    count_005 += 1
                if score >= 0.10:
                    count_010 += 1
                if score >= 0.20:
                    count_020 += 1
                if score >= 0.30:
                    count_030 += 1
            if score < CONFIDENCE_THRESHOLD or class_id != 0:
                continue
            x1 = clamp(int(round(float(row[0]) * scale_x)), 0, source_w - 1)
            y1 = clamp(int(round(float(row[1]) * scale_y)), 0, source_h - 1)
            x2 = clamp(int(round(float(row[2]) * scale_x)), 0, source_w - 1)
            y2 = clamp(int(round(float(row[3]) * scale_y)), 0, source_h - 1)
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append((x1, y1, x2 - x1, y2 - y1, score))
        self.last_max_score = max_score
        self.last_count_005 = count_005
        self.last_count_010 = count_010
        self.last_count_020 = count_020
        self.last_count_030 = count_030
        detections.sort(key=lambda item: item[4], reverse=True)
        return detections[:MAX_BOXES]


class KalmanAxis:
    """Small constant-velocity Kalman filter for one image axis."""
    def __init__(self, position):
        self.position = float(position)
        self.velocity = 0.0
        self.p00 = 25.0
        self.p01 = 0.0
        self.p10 = 0.0
        self.p11 = 100.0

    def predict(self):
        self.position += self.velocity
        p00 = self.p00 + self.p01 + self.p10 + self.p11 + KALMAN_PROCESS_POSITION
        p01 = self.p01 + self.p11
        p10 = self.p10 + self.p11
        p11 = self.p11 + KALMAN_PROCESS_VELOCITY
        self.p00, self.p01, self.p10, self.p11 = p00, p01, p10, p11

    def correct(self, measurement):
        innovation = float(measurement) - self.position
        innovation_variance = self.p00 + KALMAN_MEASUREMENT_NOISE
        position_gain = self.p00 / innovation_variance
        velocity_gain = self.p10 / innovation_variance
        p00, p01 = self.p00, self.p01
        self.position += position_gain * innovation
        self.velocity += velocity_gain * innovation
        self.p00 = (1.0 - position_gain) * p00
        self.p01 = (1.0 - position_gain) * p01
        self.p10 -= velocity_gain * p00
        self.p11 -= velocity_gain * p01


class Track:
    def __init__(self, detection):
        self.x, self.y, self.width, self.height, self.confidence = detection
        center_x = self.x + self.width / 2.0
        center_y = self.y + self.height / 2.0
        self.kalman_x = KalmanAxis(center_x)
        self.kalman_y = KalmanAxis(center_y)
        self.hits = 1
        self.misses = 0
        self.confirmed = False

    def centre(self):
        return self.kalman_x.position, self.kalman_y.position

    def velocity(self):
        return self.kalman_x.velocity, self.kalman_y.velocity

    def update_box_from_filter(self):
        frame_width, frame_height = AI_CAPTURE_SIZE
        center_x = clamp(self.kalman_x.position, 0.0, frame_width - 1.0)
        center_y = clamp(self.kalman_y.position, 0.0, frame_height - 1.0)
        self.kalman_x.position = center_x
        self.kalman_y.position = center_y
        self.x = clamp(int(round(center_x - self.width / 2.0)), 0, frame_width - self.width)
        self.y = clamp(int(round(center_y - self.height / 2.0)), 0, frame_height - self.height)

    def predict(self):
        if ENABLE_KALMAN:
            self.kalman_x.predict()
            self.kalman_y.predict()
        self.update_box_from_filter()

    def correct(self, detection):
        x, y, width, height, confidence = detection
        measured_x = x + width / 2.0
        measured_y = y + height / 2.0
        if ENABLE_KALMAN:
            self.kalman_x.correct(measured_x)
            self.kalman_y.correct(measured_y)
        else:
            self.kalman_x.position = measured_x
            self.kalman_y.position = measured_y
        alpha = EMA_ALPHA
        self.width = max(1, int(round(alpha * width + (1 - alpha) * self.width)))
        self.height = max(1, int(round(alpha * height + (1 - alpha) * self.height)))
        score_alpha = SCORE_EMA_ALPHA
        self.confidence = score_alpha * confidence + (1 - score_alpha) * self.confidence
        self.update_box_from_filter()


def detection_passes_geometry(detection):
    """Reject large or elongated regions that cannot be a competition ball."""
    _, _, width, height, _ = detection
    if width < MIN_BOX_SIDE or height < MIN_BOX_SIDE:
        return False
    aspect_ratio = width / height
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False
    frame_width, frame_height = AI_CAPTURE_SIZE
    if width > frame_width * MAX_BOX_WIDTH_RATIO:
        return False
    if height > frame_height * MAX_BOX_HEIGHT_RATIO:
        return False
    if width * height > frame_width * frame_height * MAX_BOX_AREA_RATIO:
        return False
    return True


class Tracker:
    def __init__(self):
        self.tracks = []

    def update(self, detections):
        used = [False] * len(detections)
        for track in self.tracks:
            track.predict()
            tx, ty = track.centre()
            velocity_x, velocity_y = track.velocity()
            speed = (velocity_x * velocity_x + velocity_y * velocity_y) ** 0.5
            match_distance = MATCH_DISTANCE + min(KALMAN_MAX_EXTRA_DISTANCE, int(round(speed)))
            limit = match_distance * match_distance
            best_index = -1
            best_distance = limit
            for index, detection in enumerate(detections):
                if used[index]:
                    continue
                x, y, width, height, _ = detection
                cx, cy = x + width // 2, y + height // 2
                distance = (cx - tx) * (cx - tx) + (cy - ty) * (cy - ty)
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index < 0:
                track.misses += 1
                if not track.confirmed:
                    track.hits = 0
                else:
                    track.confidence *= KALMAN_CONFIDENCE_DECAY
                continue
            used[best_index] = True
            track.correct(detections[best_index])
            track.hits += 1
            track.misses = 0
            if (
                track.hits >= CONFIRM_HITS
                and track.confidence >= DISPLAY_CONFIDENCE_THRESHOLD
            ) or (
                track.hits >= 2
                and track.confidence >= FAST_CONFIRM_THRESHOLD
            ):
                track.confirmed = True
        for index, detection in enumerate(detections):
            if not used[index]:
                self.tracks.append(Track(detection))
        self.tracks = [track for track in self.tracks if track.misses <= COAST_MAX]
        return [
            (track.x, track.y, track.width, track.height, track.confidence)
            for track in self.tracks if track.confirmed and track.misses == 0
        ]


def draw_detections(pipeline, detections, display_size, max_score):
    osd = pipeline.osd_img
    osd.clear()
    scale_x = display_size[0] / AI_CAPTURE_SIZE[0]
    scale_y = display_size[1] / AI_CAPTURE_SIZE[1]
    for index, (x, y, width, height, score) in enumerate(detections):
        dx = int(round(x * scale_x))
        dy = int(round(y * scale_y))
        dw = max(1, int(round(width * scale_x)))
        dh = max(1, int(round(height * scale_y)))
        osd.draw_rectangle(dx, dy, dw, dh, color=(0, 255, 0), thickness=2)
        osd.draw_string_advanced(
            dx, max(0, dy - 18), 18,
            "%d %d%%" % (index, int(score * 100)), color=(0, 255, 0),
        )
    osd.draw_string_advanced(
        4, 4, 20,
        "balls=%d max=%d%%" % (len(detections), int(max_score * 100)),
        color=(255, 255, 0),
    )


def send_target_coords(uart, x, y):
    if uart is None:
        return
    if x == 0xFFFF:
        uart.write(bytearray([0xAA, 0x55, 0xFF, 0xFE, 0xFF, 0xFE, 0x0D, 0x0A]))
        return
    x = int(round(x))
    y = int(round(y))
    uart.write(bytearray([0xAA, 0x55, (x >> 8) & 255, x & 255, (y >> 8) & 255, y & 255, 0x0D, 0x0A]))


def print_exception(exc):
    print("STEEL-BALL ERROR:", exc)
    try:
        sys.print_exception(exc)
    except Exception:
        pass


def main(frame_limit=None):
    pipeline = None
    detector = None
    uart = None
    if not file_exists(KMODEL_PATH):
        print("ERROR: missing model", KMODEL_PATH)
        return
    try:
        print(SCRIPT_VERSION)
        print("model=%s ai=%s display=%s" % (KMODEL_PATH, AI_CAPTURE_SIZE, DISPLAY_MODE))
        pipeline = PipeLine(
            rgb888p_size=AI_CAPTURE_SIZE,
            display_mode=DISPLAY_MODE,
            display_size=DISPLAY_SIZE,
            debug_mode=0,
        )
        pipeline.create()
        display_size = pipeline.get_display_size()
        print("stage=PIPELINE_READY display=%s" % display_size)

        detector = Yolo26Detector(KMODEL_PATH, MODEL_INPUT_SIZE, AI_CAPTURE_SIZE, 0)
        detector.config_preprocess()
        print("stage=MODEL_READY contract=[1,300,6]")

        try:
            uart = init_uart()
            print("stage=UART_READY gpio9=tx gpio10=rx baud=115200 protocol=AA55-X-Y-0D0A")
        except Exception as exc:
            print("UART disabled:", exc)
            uart = None

        tracker = Tracker()
        frame_id = 0
        while True:
            os.exitpoint()
            frame = pipeline.get_frame()
            if frame is None:
                raise RuntimeError("camera returned no frame")
            if frame_id == 0:
                print("stage=KPU_RUN_BEGIN")
            raw = detector.run(model_input(frame))
            if frame_id == 0:
                print("stage=KPU_RUN_END")
            shaped = [detection for detection in raw if detection_passes_geometry(detection)]
            stable = tracker.update(shaped) if ENABLE_TRACKING else shaped
            if frame_id == 0:
                print("stage=FIRST_FRAME_READY raw=%d stable=%d" % (len(raw), len(stable)))
            if frame_id % 30 == 0:
                print("stage=DETECTION_DIAGNOSTIC max=%.4f raw=%d shaped=%d stable=%d n005=%d n010=%d n020=%d n030=%d" % (
                    detector.last_max_score,
                    len(raw),
                    len(shaped),
                    len(stable),
                    detector.last_count_005,
                    detector.last_count_010,
                    detector.last_count_020,
                    detector.last_count_030,
                ))
            stable_max = max([detection[4] for detection in stable]) if stable else 0.0
            draw_detections(pipeline, stable, display_size, stable_max)
            pipeline.show_image()
            if frame_id % UART_SEND_EVERY_N_FRAMES == 0:
                if stable:
                    best = max(stable, key=lambda item: item[4])
                    cx = best[0] + best[2] // 2
                    cy = best[1] + best[3] // 2
                    send_target_coords(uart, cx, cy)
                else:
                    send_target_coords(uart, 0xFFFF, 0xFFFF)
            frame_id += 1
            if frame_id % 60 == 0:
                gc.collect()
            if frame_limit is not None and frame_id >= frame_limit:
                print("SMOKE_TEST_PASS frames=%d" % frame_id)
                break
    except KeyboardInterrupt:
        print("user stop")
    except BaseException as exc:
        print_exception(exc)
    finally:
        if detector is not None:
            try:
                detector.deinit()
            except Exception:
                pass
        if pipeline is not None:
            try:
                pipeline.destroy()
            except Exception:
                pass
        if uart is not None:
            try:
                uart.deinit()
            except Exception:
                pass


if __name__ == "__main__":
    main()
