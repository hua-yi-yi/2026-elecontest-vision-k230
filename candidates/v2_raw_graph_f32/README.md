# K230 钢珠模型 v2 验证包

这是一份用于定位“无钢珠场景却出现上百个框”的**诊断包**，不是可部署模型。不要与上一版模型或脚本混用。

复制：

```text
steel_ball_reference_yolo11n_1024_raw_graph_f32.kmodel
  -> /sdcard/models/steel_ball_reference_yolo11n_1024_raw_graph_f32.kmodel

steel_ball_yolo11_uart_v2.py
  -> /sdcard/steel_ball_yolo11_uart_v2.py
```

然后在 CanMV IDE K230 中运行 `/sdcard/steel_ball_yolo11_uart_v2.py`。

已知状态：该模型在 K230 上仍会出现大量假框。经转换器代码复核，模型声明为浮点输入，而 CanMV 的 YOLO11 预处理固定输出 `uint8` 图像，二者不匹配。因此它只用于输出数据诊断，不能用于实际识别。

请把启动后的完整日志、第一张无钢珠画面和第一张有钢珠画面发回；尤其关注是否打印到：

```text
stage=MODEL_READY
stage=KPU_OUTPUT_READY
stage=KPU_TENSOR
stage=KPU_SCORE
```
