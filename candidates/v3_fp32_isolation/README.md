# K230 钢珠模型 v3：未量化定位包

这不是最终部署模型，而是用来定位 v1 满屏假框来源的唯一对照包。

将以下两个文件复制到 TF 卡：

```text
steel_ball_reference_yolo11n_1024_fp32_isolation.kmodel
  -> /sdcard/models/steel_ball_reference_yolo11n_1024_fp32_isolation.kmodel

steel_ball_yolo11_uart_v3.py
  -> /sdcard/steel_ball_yolo11_uart_v3.py
```

在 CanMV IDE 中运行脚本，并在无钢珠场景观察第一帧日志：

```text
stage=KPU_OUTPUT_READY
stage=KPU_TENSOR
stage=KPU_SCORE
```

判断：若 v3 不再满屏画框，则 v1 的 PTQ 量化是根因；若 v3 仍满屏画框，则继续排查模型输出格式/CanMV 解码链路。
