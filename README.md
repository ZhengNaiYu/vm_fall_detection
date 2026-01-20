
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

## 备注

- 处理后的文件命名已统一使用 `cls{cls}`（例如：`keypoints_sequences_cls3_fps30.npy`）。
- 若修改 `fps` 或类别数，请保持 `config.yaml` 中的模板与脚本一致。

