"""K230 CanMV v1.6: detect every steel ball and publish its centre by UART2.

Required files on the TF card:
  /sdcard/models/steel_ball_reference_yolo11n_1024_uint8.kmodel
  /sdcard/steel_ball_yolo11_uart.py

This program is intentionally for the nncase 2.11 uint8-PTQ KModel produced
by this project.  Do not replace it with the former float-input
``steel_ball_320.kmodel``: that older model can prepare a tensor but hangs when
the K230 KPU starts inference on CanMV v1.6.

UART2 wiring for a standard LuShan-Pi K230 (3.3 V TTL, not RS-232 voltage):
  K230 GPIO11 -> FPIOA UART2_TXD -> receiver RX
  K230 GPIO12 -> FPIOA UART2_RXD -> receiver TX (optional for output only)
  K230 GND    -> receiver GND

UART protocol, at most five messages per second:
  BALL,N=3;120,88;251,104;382,301\r\n
The coordinate origin is the top-left of the 1024 x 1024 camera AI image. Each
``x,y`` pair is the centre of one detected ball, in the same order as its box.
"""

from libs.PipeLine import PipeLine
from libs.YOLO import YOLO11
from machine import FPIOA, UART
import gc
import os
import sys
import ulab.numpy as np


SCRIPT_VERSION = "STEEL-BALL-YOLO11-REFERENCE-V2-RAW-GRAPH"
KMODEL_PATH = "/sdcard/models/steel_ball_reference_yolo11n_1024_raw_graph_f32.kmodel"
LABELS = ["steel_ball"]
MODEL_INPUT_SIZE = [1024, 1024]
AI_CAPTURE_SIZE = [1024, 1024]
DISPLAY_MODE = "virt"       # "virt" for CanMV IDE, change to "lcd" for ST7701.
DISPLAY_SIZE = [800, 480]
CONFIDENCE_THRESHOLD = 0.20
NMS_THRESHOLD = 0.70
MAX_BOXES = 200

# A target has to appear in two nearby frames before it is published. A
# confirmed target may coast briefly through a one-frame model miss, which
# prevents the displayed box and UART coordinates from flickering.
CONFIRM_HITS = 2
COAST_MAX = 6
MATCH_DISTANCE = 72
EMA_ALPHA = 0.50

ENABLE_UART = True
UART_ID = UART.UART2
UART_BAUD = 115200
UART2_TX_GPIO = 11
UART2_RX_GPIO = 12
UART_SEND_EVERY_N_FRAMES = 10  # 30 FPS camera -> no more than 3 Hz UART output.


def file_exists(path):
    try:
        with open(path, "rb"):
            return True
    except OSError:
        return False


def init_uart():
    """Map standard-board UART2 safely and return a 115200-8N1 port."""
    if not ENABLE_UART:
        return None
    fpioa = FPIOA()
    fpioa.set_function(UART2_TX_GPIO, FPIOA.UART2_TXD)
    fpioa.set_function(UART2_RX_GPIO, FPIOA.UART2_RXD)
    # CanMV v1.6 accepts integer ``0`` for 8N1 no-parity configuration.
    return UART(UART_ID, baudrate=UART_BAUD, bits=8, parity=0, stop=1)


def model_input(frame):
    """Accept either PipeLine ndarray frames or image.Image frames."""
    if hasattr(frame, "to_numpy_ref"):
        return frame.to_numpy_ref()
    return frame


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def normalise_detections(result):
    """Convert the official YOLO11 tuple to bounded integer source boxes."""
    detections = []
    if not result or len(result) < 3:
        return detections
    boxes, class_ids, scores = result[0], result[1], result[2]
    for index in range(len(boxes)):
        if int(class_ids[index]) != 0:
            continue
        x, y, width, height = [int(round(value)) for value in boxes[index]]
        x = clamp(x, 0, AI_CAPTURE_SIZE[0] - 1)
        y = clamp(y, 0, AI_CAPTURE_SIZE[1] - 1)
        width = clamp(width, 1, AI_CAPTURE_SIZE[0] - x)
        height = clamp(height, 1, AI_CAPTURE_SIZE[1] - y)
        confidence = float(scores[index])
        detections.append((x, y, width, height, confidence))
    return detections


class Track:
    def __init__(self, x, y, width, height, confidence):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.hits = 1
        self.misses = 0
        self.confirmed = False

    def centre(self):
        return self.x + self.width // 2, self.y + self.height // 2


class Tracker:
    """Small dependency-free multi-target tracker for CanMV MicroPython."""
    def __init__(self):
        self.tracks = []

    def update(self, detections):
        used = [False] * len(detections)
        distance_squared = MATCH_DISTANCE * MATCH_DISTANCE
        for track in self.tracks:
            tx, ty = track.centre()
            best_index = -1
            best_distance = distance_squared
            for index, detection in enumerate(detections):
                if used[index]:
                    continue
                x, y, width, height, _ = detection
                cx, cy = x + width // 2, y + height // 2
                current_distance = (cx - tx) * (cx - tx) + (cy - ty) * (cy - ty)
                if current_distance < best_distance:
                    best_distance = current_distance
                    best_index = index
            if best_index < 0:
                track.misses += 1
                continue
            x, y, width, height, confidence = detections[best_index]
            used[best_index] = True
            alpha = EMA_ALPHA
            track.x = int(round(alpha * x + (1 - alpha) * track.x))
            track.y = int(round(alpha * y + (1 - alpha) * track.y))
            track.width = max(1, int(round(alpha * width + (1 - alpha) * track.width)))
            track.height = max(1, int(round(alpha * height + (1 - alpha) * track.height)))
            track.confidence = confidence
            track.hits += 1
            track.misses = 0
            if track.hits >= CONFIRM_HITS:
                track.confirmed = True
        for index, detection in enumerate(detections):
            if not used[index]:
                self.tracks.append(Track(*detection))
        self.tracks = [track for track in self.tracks if track.misses <= COAST_MAX]
        return [track for track in self.tracks if track.confirmed]


def tracks_to_detections(tracks):
    return [(track.x, track.y, track.width, track.height, track.confidence) for track in tracks]


def send_centres(uart, detections):
    """Send every centre compactly so even 100+ balls fit UART2 bandwidth."""
    if uart is None:
        return
    points = []
    for x, y, width, height, _ in detections:
        points.append("%d,%d" % (x + width // 2, y + height // 2))
    uart.write("BALL,N=%d;%s\r\n" % (len(points), ";".join(points)))


def draw_detections(pipeline, detections, display_size):
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
        4, 4, 20, "balls=%d" % len(detections), color=(255, 255, 0),
    )


def print_exception(exc):
    print("STEEL-BALL ERROR:", exc)
    try:
        sys.print_exception(exc)
    except Exception:
        pass


def print_first_kpu_tensor(detector):
    """Print the raw KPU score tensor before CanMV decodes it."""
    if not detector.results:
        print("stage=KPU_TENSOR_MISSING")
        return
    tensor = detector.results[0]
    print("stage=KPU_TENSOR shape=%s dtype=%s" % (
        getattr(tensor, "shape", None), getattr(tensor, "dtype", None),
    ))
    try:
        shape = tensor.shape
        if len(shape) == 3 and shape[1] == 5:
            scores = tensor[0][4]
            print("stage=KPU_SCORE layout=BCN max=%.6f above_020=%d" % (
                float(np.max(scores)), int(np.sum(scores > CONFIDENCE_THRESHOLD)),
            ))
        elif len(shape) == 3 and shape[2] == 5:
            scores = tensor[0][:, 4]
            print("stage=KPU_SCORE layout=BNC max=%.6f above_020=%d" % (
                float(np.max(scores)), int(np.sum(scores > CONFIDENCE_THRESHOLD)),
            ))
    except Exception as exc:
        print("stage=KPU_TENSOR_DIAGNOSTIC_ERROR", exc)


def main(frame_limit=None):
    """Run until stopped. ``frame_limit`` is for an on-board smoke test."""
    pipeline = None
    detector = None
    tracker = None
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

        detector = YOLO11(
            task_type="detect",
            mode="image",
            kmodel_path=KMODEL_PATH,
            labels=LABELS,
            rgb888p_size=AI_CAPTURE_SIZE,
            model_input_size=MODEL_INPUT_SIZE,
            conf_thresh=CONFIDENCE_THRESHOLD,
            nms_thresh=NMS_THRESHOLD,
            max_boxes_num=MAX_BOXES,
            debug_mode=0,
        )
        detector.config_preprocess()
        print("stage=MODEL_READY")

        try:
            uart = init_uart()
            uart.write("BALL,BOOT=1,FRAME=%dx%d\r\n" % (AI_CAPTURE_SIZE[0], AI_CAPTURE_SIZE[1]))
            print("stage=UART_READY gpio11=tx gpio12=rx baud=115200")
        except Exception as exc:
            # Detection must remain usable if an external UART module is not
            # connected or its pins were already reserved by another script.
            print("UART disabled:", exc)
            uart = None

        frame_id = 0
        tracker = Tracker()
        while True:
            os.exitpoint()
            frame = pipeline.get_frame()
            if frame is None:
                raise RuntimeError("camera returned no frame")
            result = detector.run(model_input(frame))
            detections = normalise_detections(result)
            stable_detections = tracks_to_detections(tracker.update(detections))
            if frame_id == 0:
                print("stage=KPU_OUTPUT_READY raw=%d stable=%d" % (len(detections), len(stable_detections)))
                print_first_kpu_tensor(detector)
            draw_detections(pipeline, stable_detections, display_size)
            pipeline.show_image()
            if frame_id % UART_SEND_EVERY_N_FRAMES == 0:
                send_centres(uart, stable_detections)
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
