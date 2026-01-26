
# Activity Detection Pipeline

基于 YOLO Pose 提取关键点，使用 LSTM/GRU/BiLSTM+Attention/Transformer 对时序关键点做动作分类（含跌倒）。核心参数集中在 `config.yaml`。

## 目录结构（简要）

```
├── config.yaml
├── yolo_pose_inference.py      # 提取关键点序列并保存 npy
├── train_pose_sequence.py      # 训练分类模型（LSTM/GRU/BiLSTMAttention/Transformer）
├── track_pose_inference.py     # 视频推理+跟踪+分类
├── export_onnx.py              # 导出分类/YOLO ONNX
├── models/                     # 权重与导出的 ONNX
├── data/
│   ├── action_videos/          # 原始动作视频（按类别子目录）
│   ├── fall_videos/            # 跌倒/正常/静止
│   └── processed/              # 关键点与标签 npy
├── sdk/                        # C++ SDK（activity_detection 可执行）
└── src/
    ├── models/                 # LSTM/GRU/BiLSTM_Attention/Transformer 定义
    ├── training/               # 训练管线
    └── utils/                  # 预处理、可视化等
```

## 环境搭建

安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：
- PyTorch
- ultralytics (YOLO)
- opencv-python
- scikit-learn
- numpy
- pyyaml

## 配置文件（config.yaml）

主要配置块：
- `train_pose_detection`：训练参数（模型类型、隐藏维度、序列长度、增强、LR 计划、输出路径模板等）。
- `yolo_pose_inference`：关键点提取（YOLO 路径、序列长度、特征模式 `rel_xy_vel`、FPS/重叠采样、保存路径）。
- `track_pose_inference`：Python 推理（输入/输出视频、YOLO 阈值、`min_box_area` 过滤小目标、模型路径/类型、类名、序列长度等）。
- `export_onnx`：导出分类或 YOLO ONNX（模型结构、checkpoint、输出路径、opset、序列长度）。

## 数据准备

1) 准备视频数据

将视频按类别放入 `data/videos/` 下的对应文件夹：

```
data/videos/
├── falls_fps{fps}/         # 跌倒视频
├── normal_fps{fps}/        # 正常活动视频
└── no_fall_static_fps{fps}/ # 静止不跌倒视频
```

文件夹名称需与 `config.yaml` 中 `yolo_pose_inference.video_folders` 配置一致。

2) 视频帧率处理（可选）

如需统一帧率，可使用 `src/utils/reduce_video_fps.py` 对视频进行重采样；或在 `yolo_pose_inference` 中开启 `auto_detect_fps`。

3) 下载 YOLO Pose 模型

从 [Ultralytics](https://github.com/ultralytics/ultralytics) 下载预训练的 YOLO Pose 模型（如 `yolo11x-pose.pt`），放入 `models/` 目录。

## 使用方法

1) 关键点提取

```bash
python yolo_pose_inference.py --config config.yaml
```

输出：`data/processed/keypoints_sequences_cls{cls}_fps{fps}.npy` 与对应标签。

2) 训练分类模型

```bash
python train_pose_sequence.py --config config.yaml
```

模型与曲线：`models/classify_cls{cls}_fps{fps}.pth`，`results/train_hist_cls{cls}_fps{fps}.png`。

3) 视频推理（Python）

```bash
python track_pose_inference.py --config config.yaml
```

可配置 YOLO 阈值、`min_box_area`（过滤远处小人）、模型类型与路径，生成带分类与跟踪 ID 的视频。

4) 导出 ONNX

```bash
python export_onnx.py --config config.yaml
# 可选覆盖：
python export_onnx.py --config config.yaml --checkpoint models/xxx.pth --output models/xxx.onnx
```

支持导出分类模型或 YOLO ONNX，依据配置选择。

## C++ SDK（sdk/）

- 可执行：`activity_detection`
- 库：`libactivity_detection_sdk.so`

### 快速开始

```bash
cd sdk
./run_inference.sh <input_video> <output_video>
```

脚本会编译（如需要）并运行，读取 `config.json`。

### 手动编译

```bash
cd sdk
rm -rf build
cmake -B build -S .
cmake --build build -j$(nproc)
```

### 处理视频/图片

```bash
cd sdk/build
./activity_detection --config_file ../config.json \
                                         --input <video_or_image> \
                                         --output <output_path>
```

支持：`.mp4 .avi .mov .mkv .wmv .jpg .jpeg .png`

### config.json 关键参数（示例）

```json
{
    "models": {
        "pose_model_path": "../../models/yolo11n-pose.onnx",
        "activity_detection_model_path": "../../models/classify_cls9_fps10.onnx"
    },
    "activity_detection": {
        "num_classes": 9,
        "class_names": ["Jump","Kick","Punch","Run","Sit","Squat","Stand","Walk","Wave"],
        "sequence_length": 10,
        "confidence_threshold": 0.7
    },
    "detection": {
        "conf_threshold": 0.2,
        "nms_threshold": 0.25,
        "min_box_area": 5000  // 过滤远处小人物；0 表示不过滤
    },
    "device": {
        "device_id": -1
    }
}
```

### 核心组件

| 模块 | 功能 | 输出 |
|------|------|------|
| PoseInferencer | YOLO11 姿态检测 | 边界框 + 17 关键点 |
| SimpleByteTracker | 多目标跟踪 | 持久化 ID |
| ActivityDetector | 时序分类（LSTM/GRU/BiLSTMAttention/Transformer） | 多类别概率与标签 |

### 调试提示

- 人太多/误检：提高 `conf_threshold`，或调大 `min_box_area`。
- 框重叠：提高 `nms_threshold`。
- 分类延迟：序列长度需填满（如 sequence_length=10，需要 10 帧后才输出分类）。

## 备注

- 处理后的文件命名已统一使用 `cls{cls}`（例如：`keypoints_sequences_cls3_fps30.npy`）。
- 若修改 `fps` 或类别数，请保持 `config.yaml` 中的模板与脚本一致。

