#include "fall_detector.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

FallDetector::FallDetector() : sequence_length_(35), num_classes_(3), confidence_threshold_(0.5f) {}

FallDetector::~FallDetector() {
    deinit();
}

bool FallDetector::init(const Config& config) {
    sequence_length_ = config.sequence_length;
    num_classes_ = config.num_classes;
    class_names_ = config.class_names;
    confidence_threshold_ = config.fall_confidence_threshold;
    
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "FallDetector");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        if (config.device_id >= 0) {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = config.device_id;
            session_options_->AppendExecutionProvider_CUDA(cuda_options);
        }
        
        session_ = std::make_unique<Ort::Session>(*env_, config.fall_detection_model_path.c_str(), *session_options_);
        
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
        
        std::cout << "Fall detection model loaded successfully" << std::endl;
        std::cout << "Number of classes: " << num_classes_ << std::endl;
        std::cout << "Sequence length: " << sequence_length_ << std::endl;
        std::cout << "Class names: ";
        for (size_t i = 0; i < class_names_.size(); ++i) {
            std::cout << class_names_[i];
            if (i < class_names_.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading fall detection model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> FallDetector::preprocessKeypoints(const std::vector<visionmatrixsdk::falldetection::KeyPoint>& keypoints) {
    std::vector<float> flattened;
    flattened.reserve(34); // 17 keypoints * 2 coordinates (x, y)
    
    for (const auto& kp : keypoints) {
        flattened.push_back(kp.x);
        flattened.push_back(kp.y);
    }
    
    // Ensure we have exactly 34 values
    while (flattened.size() < 34) {
        flattened.push_back(0.0f);
    }
    if (flattened.size() > 34) {
        flattened.resize(34);
    }
    
    return flattened;
}

void FallDetector::predict(visionmatrixsdk::falldetection::InferItem& item) {
    try {
        // Preprocess keypoints
        std::vector<float> keypoints_flat = preprocessKeypoints(item.keypoints);

        int track_id = item.id <= 0 ? 0 : item.id;
        auto& buf = keypoints_buffers_[track_id];

        // Add to buffer
        buf.push_back(keypoints_flat);

        // Maintain buffer size
        if (static_cast<int>(buf.size()) > sequence_length_) {
            buf.pop_front();
        }

        // Only predict if we have enough frames
        if (static_cast<int>(buf.size()) < sequence_length_) {
            item.fall_class = -1;
            item.fall_confidence = 0.0f;
            item.fall_name = "Buffering";
            return;
        }
        
        // Prepare input tensor
        std::vector<int64_t> input_shape = {1, sequence_length_, 34};
        std::vector<float> input_data;
        input_data.reserve(sequence_length_ * 34);

        for (const auto& frame_keypoints : buf) {
            input_data.insert(input_data.end(), frame_keypoints.begin(), frame_keypoints.end());
        }
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());
        
        // Run inference
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                    input_names_ptrs_.data(), &input_tensor, 1,
                                    output_names_ptrs_.data(), output_names_ptrs_.size());
        
        // Process output
        const float* output_data = outputs[0].GetTensorData<float>();
        auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        
        int num_classes = output_shape[1];
        
        // Apply softmax to convert logits to probabilities
        std::vector<float> probs(num_classes);
        float max_logit = output_data[0];
        for (int i = 1; i < num_classes; ++i) {
            if (output_data[i] > max_logit) max_logit = output_data[i];
        }
        
        float sum_exp = 0.0f;
        for (int i = 0; i < num_classes; ++i) {
            probs[i] = std::exp(output_data[i] - max_logit);
            sum_exp += probs[i];
        }
        for (int i = 0; i < num_classes; ++i) {
            probs[i] /= sum_exp;
        }
        
        // Store all probabilities
        item.fall_probabilities = probs;
        
        // Find class with max probability
        int max_class = 0;
        float max_prob = probs[0];
        for (int i = 1; i < num_classes; ++i) {
            if (probs[i] > max_prob) {
                max_prob = probs[i];
                max_class = i;
            }
        }
        
        item.fall_class = max_class;
        item.fall_confidence = max_prob;
        
        if (max_class >= 0 && max_class < static_cast<int>(class_names_.size())) {
            item.fall_name = class_names_[max_class];
        } else {
            item.fall_name = "Unknown";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error during fall detection: " << e.what() << std::endl;
        item.fall_class = -1;
        item.fall_confidence = 0.0f;
        item.fall_name = "Error";
    }
}

void FallDetector::reset() {
    keypoints_buffers_.clear();
}

void FallDetector::deinit() {
    session_.reset();
    session_options_.reset();
    env_.reset();
    input_names_.clear();
    output_names_.clear();
    input_names_ptrs_.clear();
    output_names_ptrs_.clear();
    keypoints_buffers_.clear();
}
