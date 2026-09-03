# K230 钢珠模型 v4：同链路 PTQ 候选

本包按已在同一块 K230 上稳定运行的 `target_best.kmodel` 转换链路生成：`uint8` 输入、`/255` 外部归一化、NoClip PTQ、按通道导出权重量化范围。

复制：

```text
steel_ball_reference_yolo11n_1024_target_pipeline_ptq.kmodel
  -> /sdcard/models/steel_ball_reference_yolo11n_1024_target_pipeline_ptq.kmodel

steel_ball_yolo11_uart_v4.py
  -> /sdcard/steel_ball_yolo11_uart_v4.py
```

在 CanMV IDE K230 中运行 `/sdcard/steel_ball_yolo11_uart_v4.py`。第一帧会打印 KPU 原始输出诊断；空场景应不再出现大量框。

这是待实机确认候选，尚不能称为最终模型。
