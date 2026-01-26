# 减少无关人士检测 - 参数调整指南

## 📊 已优化的参数

### 1. ByteTrack 跟踪器参数 ([bytetrack.yaml](bytetrack.yaml))

| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| `track_high_thresh` | 0.25 | **0.5** | 一阶匹配阈值，提高后过滤低置信度检测 |
| `track_low_thresh` | 0.1 | **0.2** | 二阶匹配阈值，减少误检恢复 |
| `new_track_thresh` | 0.25 | **0.5** | 新轨迹创建阈值，减少新ID产生 |
| `track_buffer` | 30 | **20** | 丢失轨迹保留帧数，更快删除离开的人 |

### 2. YOLO 检测参数 ([config.yaml](config.yaml))

| 参数 | 原值 | 新值 | 说明 |
|------|------|------|------|
| `conf` | 0.4 | **0.6** | YOLO置信度阈值，过滤不确定的检测 |
| `min_box_area` | - | **5000** | 最小检测框面积（像素²），过滤远处小人物 |

---

## 🎯 参数效果说明

### 提高 `conf` (YOLO置信度)
```yaml
conf: 0.6  # 原值0.4
```
- ✅ **效果**：只检测置信度>60%的人，过滤模糊、遮挡、远处的人
- ⚠️ **副作用**：可能漏掉部分有效目标
- 💡 **建议范围**：0.5-0.7（太高会漏检主要目标）

### 提高 `track_high_thresh` (跟踪匹配阈值)
```yaml
track_high_thresh: 0.5  # 原值0.25
```
- ✅ **效果**：只保留高质量的轨迹
- ⚠️ **副作用**：ID切换可能增加
- 💡 **建议范围**：0.4-0.6

### 提高 `new_track_thresh` (新轨迹阈值)
```yaml
new_track_thresh: 0.5  # 原值0.25
```
- ✅ **效果**：减少新ID的产生，避免路人甲乙丙
- ⚠️ **副作用**：新进入画面的人可能检测延迟
- 💡 **建议范围**：0.4-0.6

### 设置 `min_box_area` (最小检测框面积)
```yaml
min_box_area: 5000  # 新增参数
```
- ✅ **效果**：过滤远处的小人物（如背景路人）
- ⚠️ **计算方法**：宽×高（像素²）
- 💡 **建议值**：
  - 1080p视频：5000-10000
  - 720p视频：3000-6000
  - 不过滤：设为0

---

## 🔧 根据场景调整

### 场景1：室内单人/少人（如健身房）
推荐设置（严格模式）：
```yaml
# config.yaml
conf: 0.7              # 高置信度
min_box_area: 8000     # 过滤远处小人物

# bytetrack.yaml
track_high_thresh: 0.6
new_track_thresh: 0.6
track_buffer: 15
```

### 场景2：室外人群（如广场、街道）
推荐设置（宽松模式）：
```yaml
# config.yaml
conf: 0.5              # 适中置信度
min_box_area: 3000     # 允许较远的人

# bytetrack.yaml
track_high_thresh: 0.4
new_track_thresh: 0.4
track_buffer: 25
```

### 场景3：固定监控（主要关注画面中心）
推荐设置（中心区域优先）：
```yaml
# config.yaml
conf: 0.6
min_box_area: 6000     # 过滤边缘小人物
# 可以进一步添加位置过滤（需修改代码）
```

---

## 📈 效果验证

运行测试：
```bash
python track_pose_inference.py
```

观察输出视频，检查：
1. ✅ 无关路人是否减少
2. ✅ 主要目标是否正常跟踪
3. ✅ ID切换是否频繁
4. ⚠️ 是否有漏检情况

---

## 🛠️ 高级过滤选项（可选，需修改代码）

### 选项1：位置过滤（只关注画面中心区域）

在 `track_pose_inference.py` 中添加：
```python
# 计算检测框中心
center_x = x / W  # 归一化到0-1
center_y = y / H

# 只保留中心区域的人（例如中心50%区域）
if not (0.25 < center_x < 0.75 and 0.2 < center_y < 0.8):
    continue
```

### 选项2：持续时间过滤（只显示出现超过N帧的人）

```python
# 在主循环外添加
track_duration = {}  # pid -> frame_count

# 在检测循环中
if pid not in track_duration:
    track_duration[pid] = 0
track_duration[pid] += 1

# 只处理出现超过5帧的人
if track_duration[pid] < 5:
    continue
```

### 选项3：运动过滤（只关注运动的人，忽略静止路人）

```python
# 计算位置变化
if len(box_history[pid]) >= 2:
    prev_box = box_history[pid][-2]
    curr_box = box_history[pid][-1]
    movement = np.sqrt((curr_box[0] - prev_box[0])**2 + 
                       (curr_box[1] - prev_box[1])**2)
    
    # 过滤静止的人（移动距离<5像素）
    if movement < 5:
        continue
```

---

## 💡 调参技巧

1. **逐步调整**：一次只改一个参数，观察效果
2. **先调conf**：最直接有效的参数
3. **再调ByteTrack**：优化跟踪稳定性
4. **最后调过滤**：根据具体场景添加自定义过滤

如果调整后：
- **漏检太多** → 降低 `conf` 和 `track_high_thresh`
- **误检太多** → 提高 `conf` 和 `min_box_area`
- **ID切换频繁** → 提高 `track_buffer`，降低 `match_thresh`
- **新人出现慢** → 降低 `new_track_thresh`
