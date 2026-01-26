#ifndef ACTIVITY_DETECTOR_H
#define ACTIVITY_DETECTOR_H

#include "vmsdk.h"
#include "config_loader.h"
#include <onnxruntime_cxx_api.h>
#include <deque>
#include <memory>
#include <unordered_map>

class ActivityDetector {
public:
    ActivityDetector();
    ~ActivityDetector();
    
    bool init(const Config& config);
    void predict(visionmatrixsdk::falldetection::InferItem& item);
    void reset();
    void deinit();
    
private:
    // Convert absolute keypoints (17 x (x,y,conf)) to relative XY (34)
    std::vector<float> toRelativeXY(const std::vector<visionmatrixsdk::falldetection::KeyPoint>& keypoints);
    // Build sequence features: rel_xy (+ velocity if model expects 68 dims)
    void buildSequenceFeatures(const std::deque<std::vector<float>>& rel_xy_seq,
                               std::vector<float>& out_features) const;
    // Normalize features by max-abs over sequence
    void normalizeMaxAbs(std::vector<float>& features) const;
    
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
    int feature_dim_ = 34; // expected feature dimension from ONNX: 34 or 68
};

#endif // ACTIVITY_DETECTOR_H
