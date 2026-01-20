#!/bin/bash

# 检查是否已编译
if [ ! -f "build/fall_detection" ]; then
    echo "可执行文件不存在，开始编译..."
    ./build.sh
fi

# 使用 Python conda 环境的 ONNX Runtime 库
CONDA_ONNX_LIB="/home/u/miniconda3/lib/python3.11/site-packages/onnxruntime/capi"
ONNX_LIB="${PWD}/onnxruntime-linux-x64-1.16.3/lib"

# 优先使用 conda 环境的库
export LD_LIBRARY_PATH="${CONDA_ONNX_LIB}:${ONNX_LIB}:$LD_LIBRARY_PATH"

echo "Using ONNX Runtime libraries from: $CONDA_ONNX_LIB"

# 运行推理
cd build
LD_LIBRARY_PATH="${CONDA_ONNX_LIB}:${ONNX_LIB}:$LD_LIBRARY_PATH" ./fall_detection \
    --config_file ../config.json \
    --input ../../data/test_videos/recorded_video_20260115_111419_30fps.mp4 \
    --output ../../output_fall_detected_cpp.mp4

echo ""
echo "推理完成！输出文件: ../../output_fall_detected_cpp.mp4"
