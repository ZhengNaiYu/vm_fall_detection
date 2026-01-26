#!/bin/bash

# 检查是否已编译
if [ ! -f "build/activity_detection" ]; then
    echo "可执行文件不存在，开始编译..."
    ./build.sh
fi

# 使用 Python conda 环境的 ONNX Runtime 库
CONDA_ONNX_LIB="/home/u/miniconda3/lib/python3.11/site-packages/onnxruntime/capi"
ONNX_LIB="${PWD}/onnxruntime-linux-x64-1.16.3/lib"

# 优先使用 conda 环境的库
export LD_LIBRARY_PATH="${CONDA_ONNX_LIB}:${ONNX_LIB}:$LD_LIBRARY_PATH"

# 禁用 OpenCV/FFmpeg 硬件解码以避免 h264_bm 报错
# 参考: OPENCV_FFMPEG_CAPTURE_OPTIONS
# - hw_acceleration;none  强制关闭硬件加速
# - video_codec;h264      使用软件 h264 解码器
# - prefer_system_codecs;true 优先系统编解码器，避开定制硬件插件
export OPENCV_FFMPEG_CAPTURE_OPTIONS="hw_acceleration;none|prefer_system_codecs;true"

echo "Using ONNX Runtime libraries from: $CONDA_ONNX_LIB"

# # 选择输入/输出（可通过参数覆盖）
# INPUT_VIDEO="${1:-../../data/test_videos/aung_la.mp4}"
# OUTPUT_VIDEO="${2:-../../output_activity_detected_cpp.mp4}"

# # 基本检查：输入文件是否存在（从 build/ 目录相对路径计算）
# if [ ! -f "$INPUT_VIDEO" ]; then
#     echo "[ERROR] 输入视频不存在: $INPUT_VIDEO"
#     echo "提示: 路径是相对于 sdk/build/ 的。如果从 sdk/ 运行，此路径应以 ../.. 开头。"
#     exit 1
# fi

# 运行推理
cd build
LD_LIBRARY_PATH="${CONDA_ONNX_LIB}:${ONNX_LIB}:$LD_LIBRARY_PATH" \
OPENCV_FFMPEG_CAPTURE_OPTIONS="${OPENCV_FFMPEG_CAPTURE_OPTIONS}" \
./activity_detection \
        --config_file ../config.json \
        --input ../../data/test_videos_fps_reduced/aung_la.mp4 \
        --output ../../output_activity_detected_cpp.mp4

echo ""
echo "推理完成！输出文件: ../../output_activity_detected_cpp.mp4"