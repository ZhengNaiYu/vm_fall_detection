#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>
#include <vector>
#include <memory>

struct Config {
    // Model paths
    std::string pose_model_path;
    std::string activity_detection_model_path;
    
    // Device settings
    int device_id;
    
    // Activity detection settings
    int num_classes;
    std::vector<std::string> class_names;
    int sequence_length;
    float fall_confidence_threshold;
    
    // Detection settings
    float nms_threshold;
    float conf_threshold;
    float min_box_area;
    
    // Display settings
    int display_frequency;
    int progress_frequency;
    
    Config() : device_id(-1), num_classes(3), sequence_length(35), 
               fall_confidence_threshold(0.5f), nms_threshold(0.25f), conf_threshold(0.4f), min_box_area(0.f),
               display_frequency(1), progress_frequency(30) {}
};

class ConfigLoader {
public:
    static std::unique_ptr<Config> loadFromFile(const std::string& config_path);
    static bool validateConfig(const Config& config);
    
private:
    static std::string readFile(const std::string& file_path);
    static std::unique_ptr<Config> parseJson(const std::string& json_str);
};

#endif // CONFIG_LOADER_H
