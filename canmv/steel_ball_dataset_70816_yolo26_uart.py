"""CanMV entry point for the dataset_70816 direct-resize recall model.

Required files on the TF card:
  /sdcard/models/steel_ball_dataset_70816_yolo26n_416_stretched_recall_i16w8.kmodel
  /sdcard/steel_ball_yolo26_uart_epoch19.py
  /sdcard/steel_ball_dataset_70816_yolo26_uart.py

The model was trained on 512x288 frames physically stretched to 416x416,
matching the CanMV AI2D resize used by the shared runtime script.
"""

import steel_ball_yolo26_uart_epoch19 as app


app.SCRIPT_VERSION = "STEEL-BALL-DATASET-70816-STRETCHED-416-I16W8-KALMAN-V2"
app.KMODEL_PATH = "/sdcard/models/steel_ball_dataset_70816_yolo26n_416_stretched_recall_i16w8.kmodel"
app.MODEL_INPUT_SIZE = [416, 416]
app.CONFIDENCE_THRESHOLD = 0.20
app.DISPLAY_CONFIDENCE_THRESHOLD = 0.20
app.FAST_CONFIRM_THRESHOLD = 0.75
app.CONFIRM_HITS = 2
app.MIN_BOX_SIDE = 8
app.ENABLE_KALMAN = True
app.COAST_MAX = 4
app.KALMAN_MAX_EXTRA_DISTANCE = 48
app.KALMAN_CONFIDENCE_DECAY = 0.85


if __name__ == "__main__":
    app.main()
