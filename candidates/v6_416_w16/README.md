# K230 钢珠模型 v6：16 位权重量化候选

v5 的 416 部署路径已消除 1024 模型的大规模假框，但板端量化仍比电脑原始模型多出误报。v6 保持相同的摄像头、输入尺寸和 103 张校准图，只把权重量化精度从 8 位提升到 16 位，减少模型分数失真。

```text
steel_ball_reference_yolo11n_416_target_pipeline_calib103_w16.kmodel
  -> /sdcard/models/steel_ball_reference_yolo11n_416_target_pipeline_calib103_w16.kmodel

steel_ball_yolo11_uart_v6.py
  -> /sdcard/steel_ball_yolo11_uart_v6.py
```

v6 模型约 5.8 MB，比 v5 更大。请用与 v5 相同画面比对第一帧的 `raw=`、`above_020=` 和实际画框数。

脚本已更新为 v6.1：关闭移动镜头下的旧框滑行，使用标准 YOLO11 NMS 0.45，并在左上角同时显示 `balls=确认框数 raw=当前帧原始框数`。
