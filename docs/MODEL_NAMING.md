# 模型架构规范命名

## 📋 模型文件结构

### src/models/ 目录

```
src/models/
├── __init__.py              # 模型导出接口
├── lstm.py                  # 标准LSTM模型
├── gru.py                   # 标准GRU模型
├── bilstm_attention.py      # BiLSTM + Attention 模型
├── vae.py                   # VAE模型
├── physical_rules.py        # 物理规则检测
└── deprecated/              # 已弃用的文件
    ├── fall_detection_lstm.py     (use lstm.py instead)
    ├── fall_detection_gru.py      (use gru.py instead)
    └── improved_lstm.py           (use bilstm_attention.py instead)
```

---

## 🏗️ 模型命名规范

| 文件名 | 类名 | 描述 | 用途 |
|--------|------|------|------|
| `lstm.py` | `LSTM` | 标准LSTM | 基础时序分类 |
| `gru.py` | `GRU` | 标准GRU | 轻量级时序分类 |
| `bilstm_attention.py` | `BiLSTMAttention` | 双向LSTM+注意力 | 高精度分类 |
| `bilstm_attention.py` | `Transformer` | Transformer编码器 | 长距离依赖捕获 |

---

## 📝 使用方式

### 配置文件中指定模型

```yaml
train_pose_detection:
  type: "BiLSTMAttention"     # 选项：LSTM, GRU, BiLSTMAttention, Transformer
  hidden_size: 128
  num_layers: 2
  dropout_prob: 0.4
```

### Python代码中导入

```python
from src.models import LSTM, GRU, BiLSTMAttention, Transformer

# 创建模型
model = BiLSTMAttention(
    input_size=68,
    hidden_size=128,
    num_layers=2,
    num_classes=9,
    dropout_prob=0.4
)
```

---

## 🔄 向后兼容性

为了兼容旧的代码，模块提供了别名映射：

```python
# 这些旧的导入仍然可用（已弃用）
from src.models import FallDetectionLSTM, FallDetectionGRU, ImprovedLSTM, TransformerEncoder

# 它们会自动映射到新的命名：
FallDetectionLSTM = LSTM
FallDetectionGRU = GRU
ImprovedLSTM = BiLSTMAttention
TransformerEncoder = Transformer
```

### 旧配置兼容

配置文件中仍然支持旧的模型名称（会自动转换）：

```yaml
type: "ImprovedLSTM"        # ✅ 仍然有效（会使用BiLSTMAttention）
type: "BiLSTMAttention"     # ✅ 新的规范名称
```

---

## 📊 模型对比

| 特性 | LSTM | GRU | BiLSTMAttention | Transformer |
|------|------|-----|-----------------|-------------|
| 参数量 | ⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 推理速度 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| 精度 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 长距离依赖 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 适合场景 | 简单动作 | 轻量级 | **推荐** | 复杂动作 |

---

## 🚀 推荐配置

### 一般场景（推荐）
```yaml
train_pose_detection:
  type: "BiLSTMAttention"
  hidden_size: 128
  num_layers: 2
  dropout_prob: 0.4
```

### 数据量小或推理速度优先
```yaml
train_pose_detection:
  type: "LSTM"
  hidden_size: 64
  num_layers: 2
  dropout_prob: 0.5
```

### 高精度需求和计算能力充足
```yaml
train_pose_detection:
  type: "Transformer"
  hidden_size: 256
  num_layers: 4
  nhead: 8
  dropout_prob: 0.3
```

---

## 📖 类详解

### LSTM
标准LSTM实现，适合基础时序分类任务。
- 输入: (batch, seq_len, input_size)
- 输出: (batch, num_classes)

### GRU
GRU是LSTM的轻量级替代品，参数更少但表现相近。
- 对小数据集更友好
- 推理速度更快

### BiLSTMAttention
- 双向LSTM：捕获前后文信息
- 注意力机制：学习每帧的重要性权重
- 深分类器：增强特征提取能力
- **推荐用于动作识别**

### Transformer
- 多头自注意力：并行计算多个表示子空间
- 位置编码：保留时序信息
- 适合长序列和复杂时序依赖

---

## 🔧 训练命令

```bash
# 使用标准LSTM
python train_pose_sequence.py

# 使用BiLSTMAttention（推荐）
python train_pose_sequence.py
# 配置文件已默认使用BiLSTMAttention

# 使用Transformer
# 修改config.yaml中 type: "Transformer"
python train_pose_sequence.py
```

---

## ⚠️ 迁移指南

如果你有使用旧命名的代码：

**旧代码（仍然有效）：**
```python
from src.models import FallDetectionLSTM, ImprovedLSTM

model = ImprovedLSTM(input_size=68, hidden_size=128, ...)
```

**新代码（推荐）：**
```python
from src.models import BiLSTMAttention

model = BiLSTMAttention(input_size=68, hidden_size=128, ...)
```

两者完全等价，但推荐使用新的规范命名以提高代码可读性。
