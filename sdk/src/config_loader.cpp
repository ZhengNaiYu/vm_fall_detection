#include "config_loader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

std::unique_ptr<Config> ConfigLoader::loadFromFile(const std::string& config_path) {
    try {
        std::string json_content = readFile(config_path);
        if (json_content.empty()) {
            std::cerr << "Error: Failed to read config file: " << config_path << std::endl;
            return nullptr;
        }
        
        auto config = parseJson(json_content);
        if (!config) {
            std::cerr << "Error: Failed to parse config file: " << config_path << std::endl;
            return nullptr;
        }
        
        if (!validateConfig(*config)) {
            std::cerr << "Error: Invalid configuration" << std::endl;
            return nullptr;
        }
        
        return config;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return nullptr;
    }
}

std::string ConfigLoader::readFile(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::unique_ptr<Config> ConfigLoader::parseJson(const std::string& json_str) {
    auto config = std::make_unique<Config>();
    
    try {
        // Simple JSON parsing helper
        auto findValue = [&json_str](const std::string& key) -> std::string {
            std::string search_key = "\"" + key + "\":";
            size_t pos = json_str.find(search_key);
            if (pos == std::string::npos) return "";
            
            pos += search_key.length();
            while (pos < json_str.length() && (json_str[pos] == ' ' || json_str[pos] == '\t')) pos++;
            
            if (pos >= json_str.length()) return "";
            
            if (json_str[pos] == '"') {
                pos++;
                size_t end_pos = json_str.find('"', pos);
                if (end_pos == std::string::npos) return "";
                return json_str.substr(pos, end_pos - pos);
            } else {
                size_t end_pos = pos;
                while (end_pos < json_str.length() && 
                       (std::isdigit(json_str[end_pos]) || json_str[end_pos] == '.' || 
                        json_str[end_pos] == '-' || json_str[end_pos] == 'e' || json_str[end_pos] == 'E')) {
                    end_pos++;
                }
                return json_str.substr(pos, end_pos - pos);
            }
        };
        
        auto findArray = [&json_str](const std::string& key) -> std::vector<std::string> {
            std::vector<std::string> result;
            std::string search_key = "\"" + key + "\":";
            size_t pos = json_str.find(search_key);
            if (pos == std::string::npos) return result;
            
            pos += search_key.length();
            pos = json_str.find('[', pos);
            if (pos == std::string::npos) return result;
            
            size_t end_pos = json_str.find(']', pos);
            if (end_pos == std::string::npos) return result;
            
            std::string array_content = json_str.substr(pos + 1, end_pos - pos - 1);
            
            size_t start = 0;
            while (start < array_content.length()) {
                size_t quote_start = array_content.find('"', start);
                if (quote_start == std::string::npos) break;
                
                size_t quote_end = array_content.find('"', quote_start + 1);
                if (quote_end == std::string::npos) break;
                
                result.push_back(array_content.substr(quote_start + 1, quote_end - quote_start - 1));
                start = quote_end + 1;
            }
            
            return result;
        };
        
        // Parse model paths
        config->pose_model_path = findValue("pose_model_path");
        config->activity_detection_model_path = findValue("activity_detection_model_path");
        if (config->activity_detection_model_path.empty()) {
            // backward compatibility
            config->activity_detection_model_path = findValue("fall_detection_model_path");
        }
        
        // Parse device settings
        std::string device_id_str = findValue("device_id");
        if (!device_id_str.empty()) {
            config->device_id = std::stoi(device_id_str);
        }
        
        // Parse activity detection settings
        std::string num_classes_str = findValue("num_classes");
        if (!num_classes_str.empty()) {
            config->num_classes = std::stoi(num_classes_str);
        }
        
        config->class_names = findArray("class_names");
        
        std::string sequence_length_str = findValue("sequence_length");
        if (!sequence_length_str.empty()) {
            config->sequence_length = std::stoi(sequence_length_str);
        }
        
        std::string confidence_threshold_str = findValue("confidence_threshold");
        if (!confidence_threshold_str.empty()) {
            config->fall_confidence_threshold = std::stof(confidence_threshold_str);
        }
        
        // Parse detection settings
        std::string nms_threshold_str = findValue("nms_threshold");
        if (!nms_threshold_str.empty()) {
            config->nms_threshold = std::stof(nms_threshold_str);
        }
        
        std::string conf_threshold_str = findValue("conf_threshold");
        if (!conf_threshold_str.empty()) {
            config->conf_threshold = std::stof(conf_threshold_str);
        }

        std::string min_box_area_str = findValue("min_box_area");
        if (!min_box_area_str.empty()) {
            config->min_box_area = std::stof(min_box_area_str);
        }
        
        // Parse display settings
        std::string display_frequency_str = findValue("display_frequency");
        if (!display_frequency_str.empty()) {
            config->display_frequency = std::stoi(display_frequency_str);
        }
        
        std::string progress_frequency_str = findValue("progress_frequency");
        if (!progress_frequency_str.empty()) {
            config->progress_frequency = std::stoi(progress_frequency_str);
        }
        
        return config;
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return nullptr;
    }
}

bool ConfigLoader::validateConfig(const Config& config) {
    if (config.pose_model_path.empty()) {
        std::cerr << "Error: pose_model_path is empty" << std::endl;
        return false;
    }
    
    if (config.activity_detection_model_path.empty()) {
        std::cerr << "Error: activity_detection_model_path is empty" << std::endl;
        return false;
    }
    
    if (config.num_classes <= 0) {
        std::cerr << "Error: num_classes must be positive" << std::endl;
        return false;
    }
    
    if (config.sequence_length <= 0) {
        std::cerr << "Error: sequence_length must be positive" << std::endl;
        return false;
    }
    
    if (config.class_names.size() != static_cast<size_t>(config.num_classes)) {
        std::cerr << "Warning: class_names size (" << config.class_names.size() 
                  << ") does not match num_classes (" << config.num_classes << ")" << std::endl;
    }
    
    return true;
}
