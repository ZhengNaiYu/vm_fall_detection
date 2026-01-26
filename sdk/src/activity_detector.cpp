#include "activity_detector.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <array>
#include <deque>

ActivityDetector::ActivityDetector() : sequence_length_(35), num_classes_(3), confidence_threshold_(0.5f) {}

ActivityDetector::~ActivityDetector() {
    deinit();
}

bool ActivityDetector::init(const Config& config) {
    sequence_length_ = config.sequence_length;
    num_classes_ = config.num_classes;
    class_names_ = config.class_names;
    confidence_threshold_ = config.fall_confidence_threshold;
    
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ActivityDetector");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        if (config.device_id >= 0) {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = config.device_id;
            session_options_->AppendExecutionProvider_CUDA(cuda_options);
        }
        
        session_ = std::make_unique<Ort::Session>(*env_, config.activity_detection_model_path.c_str(), *session_options_);
        
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

        // Inspect input shape to determine expected feature dimension (34 or 68)
        if (num_input_nodes > 0) {
            auto type_info = session_->GetInputTypeInfo(0);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto input_shape = tensor_info.GetShape();
            if (!input_shape.empty()) {
                int64_t last_dim = input_shape.back();
                if (last_dim > 0) {
                    feature_dim_ = static_cast<int>(last_dim);
                } else {
                    // dynamic shape: fallback based on sequence_length; default to 68
                    feature_dim_ = 68;
                }
            }
        }
        
        size_t num_output_nodes = session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name_ptr = session_->GetOutputNameAllocated(i, allocator);
            std::string output_name(output_name_ptr.get());
            output_names_.push_back(output_name);
            output_names_ptrs_.push_back(output_names_.back().c_str());
        }
        
        std::cout << "Activity detection model loaded successfully" << std::endl;
        std::cout << "Number of classes: " << num_classes_ << std::endl;
        std::cout << "Sequence length: " << sequence_length_ << std::endl;
        std::cout << "Feature dim: " << feature_dim_ << std::endl;
        std::cout << "Class names: ";
        for (size_t i = 0; i < class_names_.size(); ++i) {
            std::cout << class_names_[i];
            if (i < class_names_.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading activity detection model: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> ActivityDetector::toRelativeXY(const std::vector<visionmatrixsdk::falldetection::KeyPoint>& keypoints) {
    // COCO keypoint indices
    const int LS = 5, RS = 6, LH = 11, RH = 12;
    // Collect points (x, y) with NaN for invalid
    std::vector<std::array<float,2>> pts(17, {NAN, NAN});
    for (size_t i = 0; i < keypoints.size() && i < 17; ++i) {
        pts[i][0] = keypoints[i].x;
        pts[i][1] = keypoints[i].y;
    }

    auto valid = [&](int idx){ return !(std::isnan(pts[idx][0]) || std::isnan(pts[idx][1])); };
    auto dist = [&](int a, int b){
        if (!valid(a) || !valid(b)) return NAN;
        float dx = pts[a][0] - pts[b][0];
        float dy = pts[a][1] - pts[b][1];
        return std::sqrt(dx*dx + dy*dy);
    };

    // Center: hips else shoulders else mean of valid
    float cx = 0.f, cy = 0.f;
    if (valid(LH) && valid(RH)) {
        cx = (pts[LH][0] + pts[RH][0]) * 0.5f;
        cy = (pts[LH][1] + pts[RH][1]) * 0.5f;
    } else if (valid(LS) && valid(RS)) {
        cx = (pts[LS][0] + pts[RS][0]) * 0.5f;
        cy = (pts[LS][1] + pts[RS][1]) * 0.5f;
    } else {
        // mean of valid points
        int count = 0;
        float sumx = 0.f, sumy = 0.f;
        for (int i = 0; i < 17; ++i) {
            if (!std::isnan(pts[i][0]) && !std::isnan(pts[i][1])) {
                sumx += pts[i][0];
                sumy += pts[i][1];
                count++;
            }
        }
        if (count > 0) {
            cx = sumx / count;
            cy = sumy / count;
        }
    }

    // Scale: max of shoulder/hip distance
    float d1 = dist(LS, RS);
    float d2 = dist(LH, RH);
    float scale = std::max(d1, d2);
    if (!std::isfinite(scale) || scale < 1e-6f) scale = 1.f;

    // Relative coords
    std::vector<float> rel;
    rel.reserve(34);
    for (int i = 0; i < 17; ++i) {
        float x = pts[i][0];
        float y = pts[i][1];
        if (std::isnan(x) || std::isnan(y)) {
            rel.push_back(0.f);
            rel.push_back(0.f);
        } else {
            rel.push_back((x - cx) / scale);
            rel.push_back((y - cy) / scale);
        }
    }
    return rel;
}

void ActivityDetector::normalizeMaxAbs(std::vector<float>& features) const {
    float max_abs = 0.f;
    for (float v : features) {
        float a = std::fabs(v);
        if (a > max_abs) max_abs = a;
    }
    if (std::isfinite(max_abs) && max_abs > 0.f) {
        for (auto& v : features) v /= max_abs;
    }
}

void ActivityDetector::buildSequenceFeatures(const std::deque<std::vector<float>>& rel_xy_seq,
                               std::vector<float>& out_features) const {
    // rel_xy_seq: length T, each 34
    int T = static_cast<int>(rel_xy_seq.size());
    if (feature_dim_ == 34) {
        // concatenate rel_xy only
        out_features.clear();
        out_features.reserve(T * 34);
        for (const auto& rel : rel_xy_seq) {
            out_features.insert(out_features.end(), rel.begin(), rel.end());
        }
    } else {
        // rel_xy + velocity (68)
        out_features.clear();
        out_features.reserve(T * 68);
        // Compute velocities: v[t] = rel[t] - rel[t-1], v[0]=0
        std::vector<float> prev(34, 0.f);
        for (int t = 0; t < T; ++t) {
            const auto& cur = rel_xy_seq[t];
            // rel
            out_features.insert(out_features.end(), cur.begin(), cur.end());
            // vel
            for (int k = 0; k < 34; ++k) {
                float v = (t > 0) ? (cur[k] - prev[k]) : 0.f;
                out_features.push_back(v);
            }
            prev = cur;
        }
        // normalize
        normalizeMaxAbs(out_features);
    }
}

void ActivityDetector::predict(visionmatrixsdk::falldetection::InferItem& item) {
    try {
        // Preprocess keypoints: to relative XY (34)
        std::vector<float> rel_xy = toRelativeXY(item.keypoints);

        int track_id = item.id <= 0 ? 0 : item.id;
        auto& buf = keypoints_buffers_[track_id];

        // Add to buffer
        buf.push_back(rel_xy);

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
        
        // Prepare input tensor with features based on model's expected feature_dim_
        std::vector<int64_t> input_shape = {1, sequence_length_, feature_dim_};
        std::vector<float> input_data;
        buildSequenceFeatures(buf, input_data);
        
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
        
        int num_classes = static_cast<int>(output_shape.size() >= 2 ? output_shape[1] : num_classes_);
        
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
        std::cerr << "Error during activity detection: " << e.what() << std::endl;
        item.fall_class = -1;
        item.fall_confidence = 0.0f;
        item.fall_name = "Error";
    }
}

void ActivityDetector::reset() {
    keypoints_buffers_.clear();
}

void ActivityDetector::deinit() {
    session_.reset();
    session_options_.reset();
    env_.reset();
    input_names_.clear();
    output_names_.clear();
    input_names_ptrs_.clear();
    output_names_ptrs_.clear();
    keypoints_buffers_.clear();
}
