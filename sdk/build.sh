#!/bin/bash

# Activity Detection SDK Build Script

set -e

echo "========================================="
echo "Activity Detection SDK Build Script"
echo "========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if ONNX Runtime exists
if [ ! -d "onnxruntime-linux-x64-1.16.3" ]; then
    echo -e "${YELLOW}ONNX Runtime not found. Downloading...${NC}"
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
    tar -xzf onnxruntime-linux-x64-1.16.3.tgz
    rm onnxruntime-linux-x64-1.16.3.tgz
    echo -e "${GREEN}ONNX Runtime downloaded and extracted.${NC}"
fi

# Create build directory
if [ -d "build" ]; then
    echo -e "${YELLOW}Removing existing build directory...${NC}"
    rm -rf build
fi

echo -e "${GREEN}Creating build directory...${NC}"
mkdir build
cd build

# Configure
echo -e "${GREEN}Configuring CMake...${NC}"
cmake -DCMAKE_BUILD_TYPE=Release ..

# Build
echo -e "${GREEN}Building project...${NC}"
make -j$(nproc)

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Executable: $(pwd)/activity_detection"
echo ""
echo "Usage:"
echo "  ./activity_detection --config_file ../config.json --input <video/image> --output <output>"
echo ""
