# K230 钢珠模型 v5：已验证规格候选

这个包不使用不稳定的 1024×1024 KModel。它复用了本机已验证的部署规格：K230 摄像头 AI 通道为 512×288，模型输入为 416×416，输出为 `[1, 5, 3549]`。

复制：

```text
steel_ball_reference_yolo11n_416_target_pipeline_calib103.kmodel
  -> /sdcard/models/steel_ball_reference_yolo11n_416_target_pipeline_calib103.kmodel

steel_ball_yolo11_uart_v5.py
  -> /sdcard/steel_ball_yolo11_uart_v5.py
```

在 CanMV IDE K230 中运行 `/sdcard/steel_ball_yolo11_uart_v5.py`。请先用手持钢珠画面验收，再测试空场景。

状态：候选版，尚待这块板子的实机结果确认。
