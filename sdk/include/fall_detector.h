#ifndef FALL_DETECTOR_H
#define FALL_DETECTOR_H

#include "vmsdk.h"
#include "config_loader.h"
#include <onnxruntime_cxx_api.h>
#include <deque>
#include <memory>
#include <unordered_map>

class FallDetector {
public:
    FallDetector();
    ~FallDetector();
    
    bool init(const Config& config);
    void predict(visionmatrixsdk::falldetection::InferItem& item);
    void reset();
    void deinit();
    
private:
    std::vector<float> preprocessKeypoints(const std::vector<visionmatrixsdk::falldetection::KeyPoint>& keypoints);
    
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> session_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_ptrs_;
    std::vector<const char*> output_names_ptrs_;
    
    // Per-track keypoint buffers to avoid mixing identities
    std::unordered_map<int, std::deque<std::vector<float>>> keypoints_buffers_;
    
    // Configuration
    int sequence_length_;
    int num_classes_;
    std::vector<std::string> class_names_;
    float confidence_threshold_;
};

#endif // FALL_DETECTOR_H
