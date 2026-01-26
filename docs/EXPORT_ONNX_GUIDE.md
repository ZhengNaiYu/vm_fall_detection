# ONNX 模型导出指南

## 📋 概述

优化后的 `export_onnx.py` 支持灵活导出多种模型到 ONNX 格式：
- ✅ 分类模型（LSTM, GRU, BiLSTMAttention, Transformer）
- ✅ YOLO 姿态检测模型
- ✅ 导出一个或两个模型

---

## 🚀 快速使用

### 导出分类模型（默认）

```bash
python export_onnx.py
```

### 导出 YOLO 模型

```bash
python export_onnx.py --model yolo
```

### 导出两个模型

```bash
python export_onnx.py --model all
```

### 指定自定义路径

```bash
python export_onnx.py \
  --checkpoint models/my_classifier.pth \
  --output models/my_classifier.onnx
```

---

## 📝 配置说明

### config.yaml - export_onnx 部分

```yaml
export_onnx:
  # 分类模型类型
  type: "BiLSTMAttention"          # LSTM | GRU | BiLSTMAttention | Transformer
  
  # 模型参数（必须与训练时一致）
  input_size: 68                   # 输入特征维度
  hidden_size: 128                 # 隐层大小
  num_layers: 2                    # LSTM/GRU 层数
  dropout_prob: 0.4                # Dropout 概率
  num_classes: 9                   # 输出类别数
  nhead: 4                         # Transformer 多头注意力头数
  sequence_length: 15              # 虚拟输入序列长度
  
  # 导出设置
  checkpoint: "models/improved_lstm_cls9_fps10.pth"    # 输入模型文件
  output: "models/improved_lstm_cls9_fps10.onnx"       # 输出 ONNX 文件
  opset_version: 17                                    # ONNX opset 版本
```

### config.yaml - yolo_pose_inference 部分

```yaml
yolo_pose_inference:
  yolo_model_path: "models/yolo11n-pose.pt"            # YOLO 输入模型
  output_onnx: "models/yolo11n-pose.onnx"              # 输出 ONNX 文件
  opset_version: 17                                    # ONNX opset 版本
```

---

## 💻 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | str | config.yaml | 配置文件路径 |
| `--model` | str | classifier | 导出模型类型：classifier, yolo, all |
| `--checkpoint` | str | 无 | 分类模型权重文件（覆盖配置） |
| `--output` | str | 无 | 分类模型输出路径（覆盖配置） |

---

## 📋 使用场景

### 场景1：导出分类模型（常用）

```bash
# 使用配置文件中的设置
python export_onnx.py

# 或指定不同的模型
python export_onnx.py \
  --checkpoint models/lstm_cls3_fps30.pth \
  --output models/lstm_cls3_fps30.onnx
```

**输出：** 
- `models/improved_lstm_cls9_fps10.onnx` （或指定的路径）

---

### 场景2：导出 YOLO 模型

```bash
python export_onnx.py --model yolo
```

**输出：**
- `models/yolo11n-pose.onnx`

---

### 场景3：一次导出两个模型

```bash
python export_onnx.py --model all
```

**输出：**
- `models/improved_lstm_cls9_fps10.onnx`
- `models/yolo11n-pose.onnx`

---

## ⚙️ 支持的分类模型类型

| 模型类型 | 特点 | 推荐 |
|---------|------|------|
| LSTM | 标准 LSTM，轻量级 | 基础应用 |
| GRU | 轻量级，参数更少 | 小数据集 |
| **BiLSTMAttention** | **双向 LSTM + 注意力** | **⭐ 推荐** |
| Transformer | 强大，但计算量大 | 高精度需求 |

---

## 🔍 常见问题

### Q1: 导出时提示 "Checkpoint not found"

**原因：** checkpoint 文件路径不存在

**解决：**
```bash
# 检查文件是否存在
ls models/improved_lstm_cls9_fps10.pth

# 或指定正确的路径
python export_onnx.py --checkpoint path/to/your/model.pth
```

### Q2: 导出的 ONNX 文件无法使用

**原因：** 模型参数不匹配

**解决：**
确保 `export_onnx` 配置中的参数与训练时相同：
- `input_size` 必须与训练的输入维度相同
- `hidden_size`, `num_layers` 必须与模型架构相同
- `num_classes` 必须与训练的类别数相同

### Q3: 想导出多个不同配置的模型

**解决：** 创建多个配置文件或使用命令行参数覆盖：

```bash
# 导出 3 分类模型
python export_onnx.py \
  --checkpoint models/lstm_cls3_fps30.pth \
  --output models/lstm_cls3_fps30.onnx

# 导出 9 分类模型
python export_onnx.py \
  --checkpoint models/improved_lstm_cls9_fps10.pth \
  --output models/improved_lstm_cls9_fps10.onnx
```

---

## 📦 生成的 ONNX 模型规格

### 分类模型 ONNX

**输入：**
- 名称: `input`
- 形状: `(batch, seq_len, input_size)` - 动态
- 数据类型: `float32`

**输出：**
- 名称: `output`
- 形状: `(batch, num_classes)`
- 数据类型: `float32`

**示例：**
```python
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
sess = ort.InferenceSession('models/improved_lstm_cls9_fps10.onnx')

# 准备输入（15 帧，68 维特征）
x = np.random.randn(1, 15, 68).astype(np.float32)

# 推理
output = sess.run(None, {'input': x})
print(output[0].shape)  # (1, 9) - 9 个类别的概率
```

### YOLO 模型 ONNX

支持动态输入尺寸，详见 [YOLO 官方文档](https://docs.ultralytics.com/modes/export/)

---

## 🛠️ 扩展开发

### 添加新的导出格式

如需导出其他格式（如 TorchScript, TensorFlow 等），修改 `export_onnx.py`：

```python
def export_torchscript(export_cfg, checkpoint, output):
    """Export to TorchScript format"""
    model = build_classifier_model(export_cfg)
    model.load_state_dict(torch.load(checkpoint))
    traced = torch.jit.trace(model, torch.randn(1, 15, 68))
    traced.save(output)
```

### 添加模型验证

```bash
# 验证导出的 ONNX 模型
python -m onnx.checker check models/improved_lstm_cls9_fps10.onnx
```

---

## 📚 相关资源

- [ONNX 官方文档](https://onnx.ai/)
- [PyTorch ONNX 导出指南](https://pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime 推理](https://onnxruntime.ai/)
- [YOLO 模型导出](https://docs.ultralytics.com/modes/export/)
