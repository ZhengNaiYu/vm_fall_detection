
# Fall Detection Pipeline

本项目基于 YOLO Pose 提取人体关键点，并使用 LSTM/GRU 对时序关键点进行跌倒检测。所有参数集中在 `config.yaml`。

## 目录结构

```
├── config.yaml
├── yolo_inference.py          # 关键点提取与预处理
├── train_fall_detection.py    # 训练 LSTM/GRU 模型
├── track_fall_inference.py    # 视频跟踪与跌倒检测
├── export_onnx.py             # 导出 ONNX
├── models/                    # 权重文件
├── data/
│   ├── videos/                # 原始视频
│   └── processed/             # 处理后的 .npy
└── src/
    └── utils/
        └── extract_keypoints_from_video.py
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

当前包含四个模块化配置块：

- `train_fall_detection`：训练相关（包含实验、模型、数据集与输出路径参数）
- `yolo_pose_inference`：关键点提取与预处理（YOLO 模型、序列长度、归一化、保存路径等）
- `track_fall_inference`：视频推理（输入/输出视频、窗口大小、是否用 LSTM、阈值等）
- `export_onnx`：ONNX 导出（模型结构、序列长度、checkpoint、opset、输出路径）

示例关键路径模板（与当前配置一致）：

- 数据集输出：`data/processed/keypoints_sequences_cls{cls}_fps{fps}.npy`
- 标签输出：`data/processed/labels_cls{cls}_fps{fps}.npy`
- 模型保存：`models/lstm_cls{cls}_fps{fps}.pth`
- 训练曲线：`results/train_hist_cls{cls}_fps{fps}.png`

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

如需统一帧率，可使用 `src/utils/reduce_video_fps.py` 对视频进行重采样。

3) 下载 YOLO Pose 模型

从 [Ultralytics](https://github.com/ultralytics/ultralytics) 下载预训练的 YOLO Pose 模型（如 `yolo11x-pose.pt`），放入 `models/` 目录。

## 使用方法

1) 关键点提取与预处理

```bash
python yolo_inference.py --config config.yaml
```

输出位于 `data/processed/`，文件名根据 `fps` 与 `num_classes` 模板生成。

2) 训练模型

```bash
python train_fall_detection.py --config config.yaml
```

读取 `train_fall_detection` 配置，训练完成后保存模型与训练曲线。

3) 视频推理（跟踪 + 跌倒检测）

```bash
python track_fall_inference.py --config config.yaml
```

读取 `track_fall_inference` 配置，生成带标注的视频输出。注意：可通过设置use_lstm参数选择是否加载之前训练好的LSTM检测模型，也可直接基于物理规则判断是否跌倒。



4) 导出 ONNX

```bash
python export_onnx.py --config config.yaml
# 可选覆盖：
python export_onnx.py --config config.yaml --checkpoint models/xxx.pth --output models/xxx.onnx
```

读取 `export_onnx` 配置，按设定的输入序列长度与 opset 导出。

## C++ SDK 使用方法

编译好的 SDK 位于 `sdk/` 目录。

### 快速开始

```bash
cd sdk
./run_inference.sh
```

该脚本自动编译（如需要）并运行推理，输出带标注的视频到 `output_fall_detected_cpp.mp4`。

### 手动编译

```bash
cd sdk
rm -rf build
cmake -B build -S .
cmake --build build -j$(nproc)
```

编译产物：
- `build/fall_detection` - 可执行程序
- `build/libfall_detection_sdk.so` - 动态链接库

### 处理视频/图片

```bash
cd sdk/build
./fall_detection --config_file ../config.json \
                 --input <video_or_image> \
                 --output <output_path>
```

支持格式：`.mp4 .avi .mov .mkv .wmv .jpg .jpeg .png`

### 配置文件（config.json）

关键参数：

```json
{
    "models": {
        "pose_model_path": "../../models/yolo11n-pose.onnx",
        "fall_detection_model_path": "../../models/lstm_cls3_fps5.onnx"
    },
    "detection": {
        "conf_threshold": 0.2,    // 姿态检测阈值（低=多人，高=精准）
        "nms_threshold": 0.25     // NMS去重叠阈值
    },
    "fall_detection": {
        "confidence_threshold": 0.7,  // 跌倒分类阈值
        "sequence_length": 35         // 序列长度（帧数）
    },
    "device": {
        "device_id": -1               // -1=CPU, ≥0=GPU设备ID
    }
}
```

### 核心组件

| 模块 | 功能 | 输出 |
|------|------|------|
| PoseInferencer | YOLO11 姿态检测 | 人体边界框 + 17个关键点 |
| SimpleByteTracker | 多目标跟踪 | 为每人分配持久化ID |
| FallDetector | LSTM跌倒分类 | 3类概率（Fall/Normal/Static） |

### 调试技巧

- **检测人数过多**：降低 `conf_threshold`
- **重叠框过多**：提高 `nms_threshold`
- **前N帧无跌倒分类**：正常，缓冲35帧数据后开始分类
- **查看库调用**：库接口定义在 `include/vmsdk.h`

## 备注

- 处理后的文件命名已统一使用 `cls{cls}`（例如：`keypoints_sequences_cls3_fps30.npy`）。
- 若修改 `fps` 或类别数，请保持 `config.yaml` 中的模板与脚本一致。

