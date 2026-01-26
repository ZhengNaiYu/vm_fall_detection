#include "vmsdk.h"
#include "pose_inferencer.h"
#include "activity_detector.h"
#include "config_loader.h"
#include "tracker.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>

namespace visionmatrixsdk
{
    namespace falldetection
    {
        struct FallDetectionModel {
            PoseInferencer* pose_detector;
            ActivityDetector* activity_detector;
            SimpleByteTracker* tracker;
            std::unique_ptr<Config> config;
            
            FallDetectionModel() : pose_detector(nullptr), activity_detector(nullptr), tracker(nullptr) {}
            ~FallDetectionModel() {
                if (pose_detector) {
                    pose_detector->deinit();
                    delete pose_detector;
                }
                if (activity_detector) {
                    activity_detector->deinit();
                    delete activity_detector;
                }
                if (tracker) {
                    delete tracker;
                }
            }
        };

        // Helper functions
        void drawPoseKeypoints(cv::Mat& frame, const std::vector<KeyPoint>& keypoints);
        void drawDetectionInfo(cv::Mat& frame, const InferItem& detection, const Config& config);
        
        int version(int mode) {
            switch (mode) {
            case 0:
                return 10; // user stories
            case 1:
                return 1;  // main version
            case 2:
                return 0;  // subversion
            case 3:
                return 0;  // minor version
            default:
                return -1;
            }
        }

        std::string info(int mode, int id) {
            std::ostringstream oss;
            switch (mode) {
            case -1:
                return std::to_string(version(3));
            case 0:
                switch (id) {
                case 1:
                    return R"({"version": {"mode": "0 for user stories, 1 for main, 2 for sub, 3 for minor"}})";
                case 2:
                    return R"({"info": {"mode": "query mode", "id": "query id"}})";
                case 3:
                    return R"({"init": {"setting": "MConfig structure with config_path"}})";
                default:
                    return R"({"help": ["version", "info", "init", "processVideo", "processImage"]})";
                }
            case 1:
                oss << version(1) << "." << version(2) << "." << version(3);
                return oss.str();
            case 2:
                return "Activity Detection SDK";
            case 3:
                return "Vision Matrix Technology";
            default:
                return "";
            }
            return "";
        }

        void *init(MConfig *setting) {
            if (setting == nullptr) {
                std::cerr << "Error: null setting" << std::endl;
                return nullptr;
            }
            if (setting->mode != 0) {
                std::cerr << "Error: setting mode error" << std::endl;
                return nullptr;
            }
            if (setting->config_path == nullptr) {
                std::cerr << "Error: config_path is null" << std::endl;
                return nullptr;
            }

            // Load configuration
            auto config = ConfigLoader::loadFromFile(setting->config_path);
            if (!config) {
                std::cerr << "Error: Failed to load configuration from " << setting->config_path << std::endl;
                return nullptr;
            }

            // Create model
            auto* model = new FallDetectionModel();
            model->config = std::move(config);

            // Initialize pose detector
            model->pose_detector = new PoseInferencer();
            if (!model->pose_detector->init(*model->config)) {
                std::cerr << "Error: Failed to initialize pose detector" << std::endl;
                delete model;
                return nullptr;
            }

            // Initialize activity detector
            model->activity_detector = new ActivityDetector();
            if (!model->activity_detector->init(*model->config)) {
                std::cerr << "Error: Failed to initialize activity detector" << std::endl;
                delete model;
                return nullptr;
            }

            // Initialize tracker
            model->tracker = new SimpleByteTracker();

            std::cout << "Activity detection SDK initialized successfully" << std::endl;
            return model;
        }

        int getBatchSize(void *model) {
            if (model == nullptr) {
                return -1;
            }
            return 1; // Currently support batch size of 1
        }

        void deinit(void *model) {
            if (model) {
                auto* fd_model = static_cast<FallDetectionModel*>(model);
                delete fd_model;
            }
        }

        int processVideo(void *model, const char *video_path, const char *output_path) {
            if (model == nullptr) {
                std::cerr << "Error: model is null" << std::endl;
                return -1;
            }

            auto* fd_model = static_cast<FallDetectionModel*>(model);
            
            cv::VideoCapture cap;
            if (!cap.open(video_path)) {
                // Fallback: force FFMPEG backend
                if (!cap.open(video_path, cv::CAP_FFMPEG)) {
                    std::cerr << "Error: Cannot open video file: " << video_path << std::endl;
                    std::cerr << "Hint: Ensure the path is correct relative to sdk/build and try disabling hardware decode.\n"
                              << "      Current OPENCV_FFMPEG_CAPTURE_OPTIONS: "
                              << (getenv("OPENCV_FFMPEG_CAPTURE_OPTIONS") ? getenv("OPENCV_FFMPEG_CAPTURE_OPTIONS") : "(unset)")
                              << std::endl;
                    return -1;
                }
            }

            int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            double fps = cap.get(cv::CAP_PROP_FPS);
            int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

            cv::VideoWriter writer;
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            writer.open(output_path, fourcc, fps, cv::Size(frame_width, frame_height));

            if (!writer.isOpened()) {
                std::cerr << "Error: Cannot open output video file: " << output_path << std::endl;
                return -1;
            }

            std::cout << "Processing video: " << video_path << std::endl;
            std::cout << "Resolution: " << frame_width << "x" << frame_height << std::endl;
            std::cout << "FPS: " << fps << std::endl;
            std::cout << "Total frames: " << total_frames << std::endl;

            cv::Mat frame;
            int frame_count = 0;
            auto start_time = std::chrono::high_resolution_clock::now();

            // Reset activity detector buffer and tracker
            fd_model->activity_detector->reset();
            if (fd_model->tracker) fd_model->tracker->reset();

            while (cap.read(frame)) {
                frame_count++;

                // Detect pose
                auto detections = fd_model->pose_detector->detect(frame, fd_model->config->conf_threshold);

                // Track and process each detection
                if (fd_model->tracker) {
                    fd_model->tracker->update(detections);
                }

                for (auto& detection : detections) {
                    // Run activity detection on keypoints (per track buffer)
                    fd_model->activity_detector->predict(detection);
                    
                    // Draw results
                    drawPoseKeypoints(frame, detection.keypoints);
                    drawDetectionInfo(frame, detection, *fd_model->config);
                }

                // Add frame info
                std::ostringstream frame_info;
                frame_info << "Frame: " << frame_count << "/" << total_frames;
                frame_info << " | Detections: " << detections.size();
                cv::putText(frame, frame_info.str(), cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

                writer.write(frame);

                // Progress report
                if (frame_count % fd_model->config->progress_frequency == 0) {
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                    double progress = (double)frame_count / total_frames * 100.0;
                    std::cout << "Progress: " << std::fixed << std::setprecision(1) << progress 
                             << "% (" << frame_count << "/" << total_frames << ") "
                             << "Elapsed: " << elapsed << "s" << std::endl;
                }
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
            
            std::cout << "Video processing completed!" << std::endl;
            std::cout << "Total time: " << total_elapsed << "s" << std::endl;
            std::cout << "Average FPS: " << (double)frame_count / total_elapsed << std::endl;
            std::cout << "Output saved to: " << output_path << std::endl;

            cap.release();
            writer.release();

            return 0;
        }

        int processImage(void *model, const char *image_path, const char *output_path) {
            if (model == nullptr) {
                std::cerr << "Error: model is null" << std::endl;
                return -1;
            }

            auto* fd_model = static_cast<FallDetectionModel*>(model);
            
            cv::Mat frame = cv::imread(image_path);
            if (frame.empty()) {
                std::cerr << "Error: Cannot read image file: " << image_path << std::endl;
                return -1;
            }

            std::cout << "Processing image: " << image_path << std::endl;

            // Detect pose
            auto detections = fd_model->pose_detector->detect(frame, fd_model->config->conf_threshold);

            // Track detections
            if (fd_model->tracker) {
                fd_model->tracker->update(detections);
            }

            // Process each detection
            for (auto& detection : detections) {
                // Run activity detection
                fd_model->activity_detector->predict(detection);
                
                // Draw results
                drawPoseKeypoints(frame, detection.keypoints);
                drawDetectionInfo(frame, detection, *fd_model->config);
            }

            // Save result
            if (!cv::imwrite(output_path, frame)) {
                std::cerr << "Error: Cannot write output image: " << output_path << std::endl;
                return -1;
            }

            std::cout << "Image processing completed!" << std::endl;
            std::cout << "Detections: " << detections.size() << std::endl;
            std::cout << "Output saved to: " << output_path << std::endl;

            return 0;
        }

        // Helper function implementations
        void drawPoseKeypoints(cv::Mat& frame, const std::vector<KeyPoint>& keypoints) {
            // COCO keypoint skeleton
            static const std::vector<std::pair<int, int>> skeleton = {
                {0, 1}, {0, 2}, {1, 3}, {2, 4},  // head
                {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},  // arms
                {5, 11}, {6, 12}, {11, 12},  // torso
                {11, 13}, {13, 15}, {12, 14}, {14, 16}  // legs
            };

            // Draw skeleton
            for (const auto& bone : skeleton) {
                if (bone.first >= static_cast<int>(keypoints.size()) || 
                    bone.second >= static_cast<int>(keypoints.size())) {
                    continue;
                }
                
                const auto& kp1 = keypoints[bone.first];
                const auto& kp2 = keypoints[bone.second];
                
                if (kp1.confidence > 0.3f && kp2.confidence > 0.3f) {
                    cv::line(frame, 
                            cv::Point(static_cast<int>(kp1.x), static_cast<int>(kp1.y)),
                            cv::Point(static_cast<int>(kp2.x), static_cast<int>(kp2.y)),
                            cv::Scalar(0, 255, 0), 2);
                }
            }

            // Draw keypoints
            for (const auto& kp : keypoints) {
                if (kp.confidence > 0.3f) {
                    cv::circle(frame, 
                              cv::Point(static_cast<int>(kp.x), static_cast<int>(kp.y)),
                              4, cv::Scalar(0, 0, 255), -1);
                }
            }
        }

        void drawDetectionInfo(cv::Mat& frame, const InferItem& detection, const Config& config) {
            // Draw bounding box
            cv::Scalar box_color;
            if (detection.fall_name == "Fall") {
                box_color = cv::Scalar(0, 0, 255);  // Red for fall
            } else if (detection.fall_name == "Normal") {
                box_color = cv::Scalar(0, 255, 0);  // Green for normal
            } else {
                box_color = cv::Scalar(255, 255, 0);  // Cyan for static
            }

            cv::rectangle(frame, 
                         cv::Point(detection.x1, detection.y1),
                         cv::Point(detection.x2, detection.y2),
                         box_color, 2);

            // Draw fall detection result
            if (detection.fall_class >= 0) {
                std::ostringstream label;
                label << "ID " << detection.id << " | "
                     << detection.fall_name << " " 
                     << std::fixed << std::setprecision(2) 
                     << (detection.fall_confidence * 100) << "%";
                
                int baseline = 0;
                cv::Size label_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 
                                                      0.6, 2, &baseline);
                
                // Draw label background
                cv::rectangle(frame,
                            cv::Point(detection.x1, detection.y1 - label_size.height - 10),
                            cv::Point(detection.x1 + label_size.width, detection.y1),
                            box_color, -1);
                
                // Draw label text
                cv::putText(frame, label.str(),
                           cv::Point(detection.x1, detection.y1 - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            } else {
                // Still show ID even when buffering/unknown
                std::ostringstream label;
                label << "ID " << detection.id << " | " << detection.fall_name;
                int baseline = 0;
                cv::Size label_size = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 
                                                      0.6, 2, &baseline);
                cv::rectangle(frame,
                             cv::Point(detection.x1, detection.y1 - label_size.height - 10),
                             cv::Point(detection.x1 + label_size.width, detection.y1),
                             box_color, -1);
                cv::putText(frame, label.str(),
                           cv::Point(detection.x1, detection.y1 - 5),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
            }
        }

        std::vector<InferItem> infer(void *model, const std::string &image, DetectConfig* config) {
            // Placeholder for base64 image inference
            std::vector<InferItem> results;
            std::cerr << "Warning: infer() not implemented. Use processImage() or processVideo() instead." << std::endl;
            return results;
        }

        std::vector<std::vector<InferItem>> inferFromImages(void *model, const std::vector<std::string> &images, DetectConfig* config) {
            // Placeholder for batch inference
            std::vector<std::vector<InferItem>> results;
            std::cerr << "Warning: inferFromImages() not implemented. Use processImage() or processVideo() instead." << std::endl;
            return results;
        }
    }
}
