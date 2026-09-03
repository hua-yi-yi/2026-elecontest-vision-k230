"""Capture K230 camera frames and save them to the data partition.

Target board: CanMV-K230 v1.1 with the official CanMV media API.
The default display is VIRT so the image can be previewed in CanMV IDE.
"""

import gc
import os
import sys
import time

from machine import FPIOA, Pin
from media.sensor import Sensor
from media.display import Display
from media.media import MediaManager


PHOTO_DIR = "/data/photos/"
COUNTER_FILE = PHOTO_DIR + "counter.txt"
IMAGE_SIZE = [512, 288]
DISPLAY_FPS = 30
JPEG_QUALITY = 90
CAPTURE_KEY_GPIO = 52
EXIT_KEY_GPIO = 53
KEY_DEBOUNCE_MS = 30


def ensure_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        # mkdir reports an error when the directory already exists.
        try:
            os.stat(path)
        except OSError:
            raise RuntimeError("cannot create capture directory: " + path)


def load_counter():
    try:
        with open(COUNTER_FILE, "r") as file:
            value = int(file.read().strip())
            return max(1, value)
    except (OSError, ValueError):
        return 1


def save_counter(value):
    try:
        with open(COUNTER_FILE, "w") as file:
            file.write(str(value))
    except OSError as exc:
        print("counter save error:", exc)


def print_exception(exc):
    print("CAMERA ERROR:", exc)
    try:
        sys.print_exception(exc)
    except Exception:
        pass


def init_keys():
    fpioa = FPIOA()
    fpioa.set_function(CAPTURE_KEY_GPIO, FPIOA.GPIO52)
    fpioa.set_function(EXIT_KEY_GPIO, FPIOA.GPIO53)
    # CanMV-K230 v1.1 has external 10 kOhm pull-ups on both active-low keys.
    capture_key = Pin(CAPTURE_KEY_GPIO, Pin.IN, pull=Pin.PULL_NONE)
    exit_key = Pin(EXIT_KEY_GPIO, Pin.IN, pull=Pin.PULL_NONE)
    return capture_key, exit_key


def main(frame_limit=None):
    ensure_dir(PHOTO_DIR)
    counter = load_counter()
    start_counter = counter
    saved_count = 0
    frame_id = 0
    camera = None
    sensor_running = False
    display_ready = False
    media_ready = False
    capture_key = None
    exit_key = None

    print("CAMERA-CAPTURE path=%s start=%d size=%dx%d" % (
        PHOTO_DIR, counter, IMAGE_SIZE[0], IMAGE_SIZE[1],
    ))

    try:
        # Official Canaan API: configure the camera before starting media.
        camera = Sensor()
        camera.reset()
        camera.set_framesize(width=IMAGE_SIZE[0], height=IMAGE_SIZE[1])
        camera.set_pixformat(Sensor.RGB565)

        # VIRT sends the preview to CanMV IDE and does not require HDMI/LCD.
        Display.init(
            Display.VIRT,
            width=IMAGE_SIZE[0],
            height=IMAGE_SIZE[1],
            fps=DISPLAY_FPS,
            to_ide=True,
        )
        display_ready = True

        MediaManager.init()
        media_ready = True
        camera.run()
        sensor_running = True
        capture_key, exit_key = init_keys()

        capture_raw = capture_key.value()
        capture_stable = capture_raw
        capture_changed_at = time.ticks_ms()
        exit_raw = exit_key.value()
        exit_stable = exit_raw
        exit_changed_at = capture_changed_at

        print("stage=SENSOR_READY api=media.sensor display=VIRT")
        print("stage=READY KEY0=capture KEY1=exit")

        while True:
            os.exitpoint()
            img = camera.snapshot()
            if img is None:
                print("snapshot returned no image")
                continue

            Display.show_image(img)
            frame_id += 1

            now = time.ticks_ms()
            current_capture_raw = capture_key.value()
            if current_capture_raw != capture_raw:
                capture_raw = current_capture_raw
                capture_changed_at = now
            elif (
                capture_raw != capture_stable
                and time.ticks_diff(now, capture_changed_at) >= KEY_DEBOUNCE_MS
            ):
                capture_stable = capture_raw
                if capture_stable == 0:
                    filename = "%sphoto_%04d.jpg" % (PHOTO_DIR, counter)
                    try:
                        img.save(filename, quality=JPEG_QUALITY)
                        saved_count += 1
                        counter += 1
                        save_counter(counter)
                        print("saved: %s" % filename)
                    except Exception as exc:
                        print("save error:", exc)

            current_exit_raw = exit_key.value()
            if current_exit_raw != exit_raw:
                exit_raw = current_exit_raw
                exit_changed_at = now
            elif (
                exit_raw != exit_stable
                and time.ticks_diff(now, exit_changed_at) >= KEY_DEBOUNCE_MS
            ):
                exit_stable = exit_raw
                if exit_stable == 0:
                    print("exit key pressed")
                    break

            if frame_id % 60 == 0:
                gc.collect()

            if frame_limit is not None and frame_id >= frame_limit:
                break

        print("CAPTURE_DONE frames=%d photos=%d" % (frame_id, saved_count))

    except KeyboardInterrupt:
        print("user stop")
    except BaseException as exc:
        print_exception(exc)
    finally:
        if counter != start_counter:
            save_counter(counter)
        if sensor_running:
            try:
                camera.stop()
            except Exception as exc:
                print("sensor stop error:", exc)
        if display_ready:
            try:
                Display.deinit()
            except Exception as exc:
                print("display deinit error:", exc)
        if media_ready:
            try:
                MediaManager.deinit()
            except Exception as exc:
                print("media deinit error:", exc)
        try:
            os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
            time.sleep_ms(100)
        except Exception:
            pass


if __name__ == "__main__":
    main()
