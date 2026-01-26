# Activity Detection SDK (C++)

C++ 版本的活动检测 SDK，使用 ONNX Runtime 进行推理。

## 功能特性

- **姿态检测**: 使用 YOLO11-Pose 模型检测人体关键点
- **活动检测**: 使用 LSTM 模型基于姿态序列进行动作分类
- **实时处理**: 支持视频和图像输入
- **高性能**: C++ 实现，支持 CPU 和 GPU 推理

## 环境要求

### 依赖库

- **CMake**: >= 3.10
- **C++ 编译器**: 支持 C++17 (GCC >= 7.0, Clang >= 5.0)
- **OpenCV**: >= 4.0
- **ONNX Runtime**: >= 1.16.0

### 安装依赖

#### Ubuntu/Debian

```bash
# 安装基础工具
sudo apt-get update
sudo apt-get install build-essential cmake git

# 安装 OpenCV
sudo apt-get install libopencv-dev

# 下载 ONNX Runtime
cd sdk
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
```

## 项目结构

```
sdk/
├── include/                    # 头文件
│   ├── vmsdk.h                # SDK 主接口
│   ├── config_loader.h        # 配置加载器
│   ├── pose_inferencer.h      # 姿态检测器
│   └── activity_detector.h    # 活动/动作检测器
├── src/                       # 源文件
│   ├── vmsdk.cpp
│   ├── config_loader.cpp
│   ├── pose_inferencer.cpp
│   ├── fall_detector.cpp
│   └── main.cpp               # 主程序
├── config.json                # 配置文件
├── CMakeLists.txt             # CMake 构建配置
└── README.md                  # 本文档
```

## 编译

```bash
cd sdk

# 创建构建目录
mkdir build
cd build

# 配置和编译
cmake ..
make -j$(nproc)

# 可执行文件生成在 build/ 目录下
ls -l activity_detection
```

## 配置

编辑 `config.json` 文件：

```json
{
    "models": {
        "pose_model_path": "../models/yolo11x-pose.onnx",
        "fall_detection_model_path": "../models/lstm_cls3_fps30.onnx"
    },
    "device": {
        "device_id": -1
    },
    "fall_detection": {
        "num_classes": 3,
        "class_names": ["Fall", "Normal", "Static"],
        "sequence_length": 35,
        "confidence_threshold": 0.5
    },
    "detection": {
        "nms_threshold": 0.25,
        "conf_threshold": 0.4
    },
    "display": {
        "display_frequency": 1,
        "progress_frequency": 30
    }
}
```

### 配置说明

- **models**: 模型文件路径
  - `pose_model_path`: YOLO 姿态检测模型 (.onnx)
    - `fall_detection_model_path`: LSTM 活动/动作检测模型 (.onnx)
  
- **device**: 设备配置
  - `device_id`: GPU 设备 ID，-1 表示使用 CPU
  
- **fall_detection**: 跌倒检测参数
  - `num_classes`: 分类数量
  - `class_names`: 类别名称
  - `sequence_length`: 序列长度（帧数）
  - `confidence_threshold`: 置信度阈值
  
- **detection**: 检测参数
  - `nms_threshold`: NMS 阈值
  - `conf_threshold`: 置信度阈值
  
- **display**: 显示参数
  - `display_frequency`: 显示频率
  - `progress_frequency`: 进度报告频率

## 使用方法

### 命令行参数

```bash
./activity_detection --config_file <config.json> --input <input_file> --output <output_file>
```

### 处理视频

```bash
./activity_detection \
    --config_file ../config.json \
    --input ../data/test_videos/test_video.mp4 \
    --output output_result.mp4
```

### 处理图像

```bash
./activity_detection \
    --config_file ../config.json \
    --input test_image.jpg \
    --output result_image.jpg
```

### 示例

```bash
cd build

# 处理测试视频
./activity_detection \
    --config_file ../config.json \
    --input ../../data/test_videos/recorded_video_20260115_111419_30fps.mp4 \
    --output activity_detection_result.mp4

# 处理图像
./activity_detection \
    --config_file ../config.json \
    --input test.jpg \
    --output result.jpg
```

## 导出 ONNX 模型

如果还没有 ONNX 模型，需要先从 PyTorch 模型导出：

```bash
cd ..  # 回到 fall_detection 根目录

# 导出 LSTM 模型
python export_onnx.py

# YOLO 模型导出
from ultralytics import YOLO
model = YOLO('models/yolo11x-pose.pt')
model.export(format='onnx')
```

## API 使用示例

```cpp
#include "vmsdk.h"

int main() {
    // 初始化配置
    visionmatrixsdk::falldetection::MConfig config;
    config.config_path = "config.json";
    config.mode = 0;
    
    // 初始化模型
    void* model = visionmatrixsdk::falldetection::init(&config);
    if (!model) {
        std::cerr << "Failed to initialize model" << std::endl;
        return -1;
    }
    
    // 处理视频
    int result = visionmatrixsdk::falldetection::processVideo(
        model, 
        "input.mp4", 
        "output.mp4"
    );
    
    // 清理
    visionmatrixsdk::falldetection::deinit(model);
    
    return result;
}
```

## 性能优化

### CPU 优化

- 确保使用 Release 模式编译：
  ```bash
  cmake -DCMAKE_BUILD_TYPE=Release ..
  ```

### GPU 加速

1. 安装 CUDA 和 cuDNN
2. 下载支持 CUDA 的 ONNX Runtime
3. 在 `config.json` 中设置 `device_id` 为 GPU ID（例如 0）

## 故障排除

### 找不到 ONNX Runtime

```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

或者在 CMakeLists.txt 中正确设置 `ONNXRUNTIME_ROOT` 路径。

### OpenCV 版本问题

确保 OpenCV 版本 >= 4.0：

```bash
pkg-config --modversion opencv4
```

### 模型路径错误

检查 `config.json` 中的路径是否正确，路径应该相对于可执行文件的位置。

## 输出说明

程序会在视频/图像中：
- 绘制人体骨骼关键点
- 显示边界框
- 显示跌倒检测结果（类别和置信度）
- 显示帧数和检测统计信息

边界框颜色：
- 🔴 **红色**: 检测到跌倒
- 🟢 **绿色**: 正常状态
- 🔵 **青色**: 静止状态

## 许可证

请参考项目根目录的 LICENSE 文件。

## 联系方式

如有问题或建议，请提交 Issue。
