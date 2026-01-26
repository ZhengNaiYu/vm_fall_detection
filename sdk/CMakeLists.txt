cmake_minimum_required(VERSION 3.10)
project(activity_detection_sdk)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# Set ONNXRuntime path - adjust this based on your installation
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.16.3")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Library source files
set(LIB_SOURCES
    src/vmsdk.cpp
    src/pose_inferencer.cpp
    src/activity_detector.cpp
    src/tracker.cpp
    src/config_loader.cpp
)

# Create shared library
add_library(activity_detection_sdk SHARED ${LIB_SOURCES})

# Link libraries to shared library
target_link_libraries(activity_detection_sdk 
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
)

# Set rpath for shared library
set_target_properties(activity_detection_sdk PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Create executable - links directly with source files
add_executable(activity_detection src/main.cpp ${LIB_SOURCES})

# Link libraries to executable
target_link_libraries(activity_detection 
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so.1.16.3
)

# Set rpath for executable to find ONNX Runtime libs
set_target_properties(activity_detection PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    LINK_FLAGS "-Wl,-rpath,${ONNXRUNTIME_LIB_DIR}"
)

# Installation rules (optional)
install(TARGETS activity_detection_sdk activity_detection
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/activity_detection_sdk
    FILES_MATCHING PATTERN "*.h"
)
