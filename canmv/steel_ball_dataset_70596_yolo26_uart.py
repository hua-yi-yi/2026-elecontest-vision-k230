"""CanMV entry point for the dataset_70596 YOLO26n 416 I16W8 model.

Copy this file, ``steel_ball_yolo26_uart_epoch19.py``, and the matching KModel
to the SD card. This entry point reuses the project's tested tracking and UART
implementation while selecting the newly trained model.
"""

import steel_ball_yolo26_uart_epoch19 as app


app.SCRIPT_VERSION = "STEEL-BALL-DATASET-70596-YOLO26N-416-HARDNEG-I16W8-V1"
app.KMODEL_PATH = "/sdcard/models/steel_ball_dataset_70596_yolo26n_416_hardneg_i16w8.kmodel"
app.MODEL_INPUT_SIZE = [416, 416]
app.CONFIDENCE_THRESHOLD = 0.15
app.DISPLAY_CONFIDENCE_THRESHOLD = 0.30
# Test-set balls remain above 17 pixels after scaling to the 512x288 camera,
# while the perforated-panel false positives remain below 9 pixels.
app.MIN_BOX_SIDE = 12


if __name__ == "__main__":
    app.main()
