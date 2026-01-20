#ifndef POSE_INFERENCER_H
#define POSE_INFERENCER_H

#include "vmsdk.h"
#include "config_loader.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <memory>

class PoseInferencer {
public:
    PoseInferencer();
    ~PoseInferencer();
    
    bool init(const Config& config);
    std::vector<visionmatrixsdk::falldetection::InferItem> detect(const cv::Mat& frame, float conf_threshold = 0.4f);
    void deinit();
    
private:
    cv::Mat preprocessImage(const cv::Mat& frame);
    std::vector<visionmatrixsdk::falldetection::InferItem> postprocess(
        const std::vector<Ort::Value>& outputs, 
        float scale_x, float scale_y, float conf_threshold);
    
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::Session> session_;
    
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<const char*> input_names_ptrs_;
    std::vector<const char*> output_names_ptrs_;
    
    int input_width_;
    int input_height_;
    float nms_threshold_;
};

#endif // POSE_INFERENCER_H
