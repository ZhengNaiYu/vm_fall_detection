# 提升动作识别准确率 - 优化指南

## 📌 问题诊断

当前训练准确率只有11-21%（9类随机应该是11%），原因是：
1. **关键Bug**：特征提取时没有真正应用相对坐标转换
2. 模型可能欠拟合或过拟合
3. 数据量可能不足

---

## 🔧 解决方案（按优先级）

### ⭐ 第一步：立即执行（必须）

**修复bug后重新提取和训练**

```bash
# 1. 重新提取特征（使用修复后的相对坐标转换）
python yolo_pose_inference.py

# 2. 使用原始配置训练
python train_pose_sequence.py

# 预期：准确率应该从11%提升到40-70%
```

**验证修复是否生效：**
- 检查训练loss是否正常下降
- 验证准确率是否超过30%
- 如果还是很差，继续下一步

---

### ⭐⭐ 第二步：使用改进模型（推荐）

如果第一步后准确率仍不够（<60%），使用改进的模型：

```bash
# 使用ImprovedLSTM（BiLSTM + 注意力机制）
python train_pose_sequence.py --config config.yaml
```

**改进点：**
- ✅ BiLSTM：双向捕获时序信息
- ✅ 注意力机制：学习重要帧权重
- ✅ 更大的hidden_size（64→128）
- ✅ 更合理的dropout（0.6→0.4）
- ✅ 更深的分类器

**预期：** 准确率提升5-15%

---

### ⭐⭐⭐ 第三步：高级优化（可选）

#### 选项A：使用Transformer模型

```yaml
# 修改 config.yaml
train_pose_detection:
  type: "Transformer"  # 改为Transformer
  hidden_size: 128
  num_layers: 3        # Transformer建议3-4层
  nhead: 4             # 注意力头数
```

**优势：**
- 更强的长距离依赖建模
- 并行计算更快
- 适合复杂动作序列

**适用场景：**
- 动作持续时间长
- 有明显的阶段性特征
- 数据量充足（>5000样本）

---

#### 选项B：数据增强

在 `config.yaml` 中启用：

```yaml
train_pose_detection:
  # 数据增强
  use_augmentation: true
  aug_noise_std: 0.01        # 高斯噪声
  aug_time_stretch: 0.1      # 时间拉伸±10%
```

**效果：**
- 增加数据多样性
- 提升泛化能力
- 减少过拟合

---

#### 选项C：调整超参数

**如果过拟合（训练acc高，验证acc低）：**
```yaml
dropout_prob: 0.5          # 增加dropout
weight_decay: 1e-4         # 增加正则化
batch_size: 16             # 减小batch size
```

**如果欠拟合（训练acc低）：**
```yaml
hidden_size: 256           # 增加模型容量
num_layers: 3              # 增加层数
learning_rate: 0.002       # 提高学习率
dropout_prob: 0.3          # 减少dropout
```

---

## 📊 评估指标

训练后查看 `results/` 目录：

1. **训练曲线图** - `train_hist_cls9_fps10.png`
   - Loss应该平滑下降
   - Train/Val accuracy差距<10%为正常

2. **混淆矩阵** - 查看哪些类别容易混淆
   - 如果某些类别准确率特别低，考虑增加该类数据

3. **测试集准确率**
   - >60%：基本可用
   - >70%：良好
   - >80%：优秀

---

## 🎯 推荐执行顺序

```bash
# 步骤1：修复bug重新训练（必须）
python yolo_pose_inference.py
python train_pose_sequence.py

# 步骤2：如果准确率<60%，使用改进模型
python train_pose_sequence.py --config config.yaml

# 步骤3：如果还不够，尝试Transformer
# 编辑 config.yaml，设置 type: "Transformer"
python train_pose_sequence.py --config config.yaml

# 步骤4：测试推理效果
python track_pose_inference.py --video data/test_videos/xxx.mp4
```

---

## ⚠️ 常见问题

### Q1: 训练时loss不下降？
```bash
# 检查特征是否正确提取
import numpy as np
data = np.load('data/processed/keypoints_sequences_cls9_fps10.npy')
print(data.shape)  # 应该是 (N, 15, 68)
print(data.min(), data.max())  # 应该在 -5到5之间（相对坐标）

# 如果范围很大（>100），说明相对坐标没生效
```

### Q2: 显存不够？
```yaml
batch_size: 16  # 减小batch size
hidden_size: 64  # 减小hidden size
```

### Q3: 某些类别准确率特别低？
- 检查该类别的视频数量
- 考虑使用 focal loss 处理类别不平衡
- 增加该类别的数据

---

## 📝 模型对比

| 模型 | 准确率 | 训练速度 | 参数量 | 适用场景 |
|------|--------|----------|--------|----------|
| LSTM | 基准 | 快 | 小 | 简单动作 |
| ImprovedLSTM | +5-15% | 中等 | 中等 | 推荐使用 |
| Transformer | +10-20% | 慢 | 大 | 复杂动作、数据充足 |

---

## 💡 其他建议

1. **检查数据质量**
   - 视频是否清晰
   - 人体是否完整可见
   - 动作是否标准

2. **增加数据量**
   - 每个类别至少100个视频
   - 考虑数据增强

3. **调整序列长度**
   - 快速动作：sequence_length=10
   - 慢速动作：sequence_length=20-30

4. **特征工程**
   - 添加角度特征（关节角度）
   - 添加加速度特征
   - 添加骨骼长度比例
