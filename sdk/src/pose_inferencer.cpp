#include "pose_inferencer.h"
#include <iostream>
#include <algorithm>
#include <cmath>

PoseInferencer::PoseInferencer() : input_width_(640), input_height_(640), nms_threshold_(0.25f) {}

PoseInferencer::~PoseInferencer() {
    deinit();
}

bool PoseInferencer::init(const Config& config) {
    nms_threshold_ = config.nms_threshold;
    
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "PoseInferencer");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        if (config.device_id >= 0) {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = config.device_id;
            session_options_->AppendExecutionProvider_CUDA(cuda_options);
        }
        
        session_ = std::make_unique<Ort::Session>(*env_, config.pose_model_path.c_str(), *session_options_);
        
        Ort::AllocatorWithDefaultOptions allocator;
        
        input_names_.clear();
        output_names_.clear();
        
        size_t num_input_nodes = session_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name_ptr = session_->GetInputNameAllocated(i, allocator);
            std::string input_name(input_name_ptr.get());
            input_names_.push_back(input_name);
            input_names_ptrs_.push_back(input_names_.back().c_str());
        }
        
        size_t num_output_nodes = session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name_ptr = session_->GetOutputNameAllocated(i, allocator);
            std::string output_name(output_name_ptr.get());
            output_names_.push_back(output_name);
            output_names_ptrs_.push_back(output_names_.back().c_str());
        }
        
        std::cout << "Pose detection model loaded successfully" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading pose model: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat PoseInferencer::preprocessImage(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(input_width_, input_height_));
    
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    rgb.convertTo(rgb, CV_32F, 1.0/255.0);
    
    return rgb;
}

std::vector<visionmatrixsdk::falldetection::InferItem> PoseInferencer::detect(const cv::Mat& frame, float conf_threshold) {
    std::vector<visionmatrixsdk::falldetection::InferItem> detections;
    
    try {
        cv::Mat input_image = preprocessImage(frame);
        
        float scale_x = static_cast<float>(frame.cols) / input_width_;
        float scale_y = static_cast<float>(frame.rows) / input_height_;
        
        std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
        std::vector<float> input_data(3 * input_height_ * input_width_);
        
        // Convert HWC to CHW
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < input_height_; ++h) {
                for (int w = 0; w < input_width_; ++w) {
                    input_data[c * input_height_ * input_width_ + h * input_width_ + w] = 
                        input_image.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());
        
        auto outputs = session_->Run(Ort::RunOptions{nullptr}, 
                                    input_names_ptrs_.data(), &input_tensor, 1,
                                    output_names_ptrs_.data(), output_names_ptrs_.size());
        
        detections = postprocess(outputs, scale_x, scale_y, conf_threshold);
        
    } catch (const std::exception& e) {
        std::cerr << "Error during pose detection: " << e.what() << std::endl;
    }
    
    return detections;
}

std::vector<visionmatrixsdk::falldetection::InferItem> PoseInferencer::postprocess(
    const std::vector<Ort::Value>& outputs, 
    float scale_x, float scale_y, float conf_threshold) {
    
    std::vector<visionmatrixsdk::falldetection::InferItem> results;
    
    if (outputs.empty()) {
        return results;
    }
    
    const float* output_data = outputs[0].GetTensorData<float>();
    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    
    // YOLO11 output format is TRANSPOSED: [batch, attributes, num_boxes]
    // attributes = 56 = 4 (bbox) + 1 (conf) + 17*3 (keypoints x,y,conf)
    int num_attrs = output_shape[1];
    int num_boxes = output_shape[2];
    
    int num_keypoints = (num_attrs - 5) / 3;
    
    for (int i = 0; i < num_boxes; ++i) {
        // In transposed format: data[attr_idx * num_boxes + box_idx]
        float x_center = output_data[0 * num_boxes + i];
        float y_center = output_data[1 * num_boxes + i];
        float width = output_data[2 * num_boxes + i];
        float height = output_data[3 * num_boxes + i];
        float confidence = output_data[4 * num_boxes + i];
        
        if (confidence < conf_threshold) {
            continue;
        }
        
        visionmatrixsdk::falldetection::InferItem item;
        
        // Scale back to original image size
        item.x1 = static_cast<int>((x_center - width / 2) * scale_x);
        item.y1 = static_cast<int>((y_center - height / 2) * scale_y);
        item.x2 = static_cast<int>((x_center + width / 2) * scale_x);
        item.y2 = static_cast<int>((y_center + height / 2) * scale_y);
        item.w = item.x2 - item.x1;
        item.h = item.y2 - item.y1;
        item.sim = confidence;
        item.class0 = 0; // person class
        
        // Extract keypoints (transposed format)
        for (int k = 0; k < num_keypoints && k < 17; ++k) {
            int attr_x = 5 + k * 3 + 0;
            int attr_y = 5 + k * 3 + 1;
            int attr_c = 5 + k * 3 + 2;
            
            float kp_x = output_data[attr_x * num_boxes + i] * scale_x;
            float kp_y = output_data[attr_y * num_boxes + i] * scale_y;
            float kp_conf = output_data[attr_c * num_boxes + i];
            
            item.keypoints.emplace_back(kp_x, kp_y, kp_conf);
        }
        
        results.push_back(item);
    }
    
    // Apply NMS to remove overlapping boxes
    if (results.empty()) {
        return results;
    }
    
    // Sort by confidence (descending)
    std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
        return a.sim > b.sim;
    });
    
    std::vector<visionmatrixsdk::falldetection::InferItem> nms_results;
    std::vector<bool> suppressed(results.size(), false);
    
    for (size_t i = 0; i < results.size(); ++i) {
        if (suppressed[i]) continue;
        
        nms_results.push_back(results[i]);
        
        // Suppress overlapping boxes
        for (size_t j = i + 1; j < results.size(); ++j) {
            if (suppressed[j]) continue;
            
            // Calculate IoU
            int x1 = std::max(results[i].x1, results[j].x1);
            int y1 = std::max(results[i].y1, results[j].y1);
            int x2 = std::min(results[i].x2, results[j].x2);
            int y2 = std::min(results[i].y2, results[j].y2);
            
            int inter_w = std::max(0, x2 - x1);
            int inter_h = std::max(0, y2 - y1);
            int inter_area = inter_w * inter_h;
            
            int area_i = results[i].w * results[i].h;
            int area_j = results[j].w * results[j].h;
            int union_area = area_i + area_j - inter_area;
            
            float iou = (union_area > 0) ? static_cast<float>(inter_area) / union_area : 0.0f;
            
            if (iou > nms_threshold_) {
                suppressed[j] = true;
            }
        }
    }
    
    return nms_results;
}

void PoseInferencer::deinit() {
    session_.reset();
    session_options_.reset();
    env_.reset();
    input_names_.clear();
    output_names_.clear();
    input_names_ptrs_.clear();
    output_names_ptrs_.clear();
}
