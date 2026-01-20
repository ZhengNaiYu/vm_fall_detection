#ifndef VMSDK_H
#define VMSDK_H

#include <string>
#include <vector>
#include "config_loader.h"

namespace visionmatrixsdk
{
    namespace falldetection
    {
        struct KeyPoint
        {
            KeyPoint() : x(0.0f), y(0.0f), confidence(0.0f) {}
            KeyPoint(float x_, float y_, float conf) : x(x_), y(y_), confidence(conf) {}
            float x, y;
            float confidence;
        };

        struct InferItem
        {
            InferItem()
                : mode(0), id(0), x1(0), y1(0), x2(0), y2(0), x3(0), y3(0), x4(0), y4(0), 
                  w(0), h(0), class0(0), sim(0.0f), entity(""), fall_class(-1), 
                  fall_confidence(0.0f), fall_name("") {}
            
            int mode;
            int id;
            int x1, y1;  // bounding box top-left
            int x2, y2;  // bounding box bottom-right
            int x3, y3;  // reserved
            int x4, y4;  // reserved
            int w, h;    // width and height
            std::string entity;
            int class0;  // person class (always 0)
            float sim;   // pose detection confidence
            
            // Fall detection results
            int fall_class;                          // fall class index
            float fall_confidence;                   // fall confidence
            std::string fall_name;                   // fall class name
            std::vector<float> fall_probabilities;   // all class probabilities
            
            // Pose keypoints (17 keypoints for COCO format)
            std::vector<KeyPoint> keypoints;
        };

        struct MConfig
        {
            MConfig()
                : config_path(nullptr), mode(0) {}
            char *config_path;      // configuration file path
            int mode;               // model mode
        };

        struct DetectConfig
        {
            DetectConfig()
                : batch_size(1) {}
            int n_images;
            int input_w;
            int input_h;
            int* orig_w;
            int* orig_h;
            int batch_size;
        };

        // API functions
        int version(int mode = 0);
        std::string info(int mode = 0, int id = 0);
        void *init(MConfig *setting);
        int getBatchSize(void *model);
        std::vector<InferItem> infer(void *model, const std::string &image, DetectConfig* config);
        std::vector<std::vector<InferItem>> inferFromImages(void *model, const std::vector<std::string> &images, DetectConfig* config);
        void deinit(void *model);
        
        // Extended functions for video/image processing
        int processVideo(void *model, const char *video_path, const char *output_path);
        int processImage(void *model, const char *image_path, const char *output_path);
    }
}

#endif // VMSDK_H
