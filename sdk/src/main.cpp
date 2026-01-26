#include <iostream>
#include <fstream>
#include <algorithm>
#include "vmsdk.h"

void printUsage() {
    std::cout << "Usage: ./activity_detection [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --config_file PATH    Path to configuration JSON file (required)" << std::endl;
    std::cout << "  --input PATH          Input file path (image or video)" << std::endl;
    std::cout << "  --output PATH         Output file path" << std::endl;
    std::cout << "  --help               Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  ./activity_detection --config_file config.json --input input.jpg --output result.jpg" << std::endl;
    std::cout << "  ./activity_detection --config_file config.json --input input.mp4 --output result.mp4" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage();
        return 1;
    }
    
    std::string config_file;
    std::string input_file;
    std::string output_file = "result";
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage();
            return 0;
        } else if (arg == "--config_file" && i + 1 < argc) {
            config_file = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }
    
    // Validate required arguments
    if (config_file.empty()) {
        std::cerr << "Error: Configuration file not specified" << std::endl;
        printUsage();
        return 1;
    }
    
    if (input_file.empty()) {
        std::cerr << "Error: Input file not specified" << std::endl;
        printUsage();
        return 1;
    }
    
    // Check if files exist
    std::ifstream config_check(config_file);
    if (!config_check.good()) {
        std::cerr << "Error: Configuration file not found: " << config_file << std::endl;
        return 1;
    }
    
    std::ifstream input_check(input_file);
    if (!input_check.good()) {
        std::cerr << "Error: Input file not found: " << input_file << std::endl;
        return 1;
    }
    
    try {
        // Initialize SDK with configuration file
        visionmatrixsdk::falldetection::MConfig config;
        config.config_path = const_cast<char*>(config_file.c_str());
        config.mode = 0;
        
        void* model = visionmatrixsdk::falldetection::init(&config);
        if (!model) {
            std::cerr << "Error: Failed to initialize model" << std::endl;
            return 1;
        }
        
        // Determine file type and set output extension if not specified
        std::string extension = input_file.substr(input_file.find_last_of("."));
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
        
        // Auto-generate output filename if no extension provided
        if (output_file.find('.') == std::string::npos) {
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
                output_file += ".jpg";
            } else {
                output_file += ".mp4";
            }
        }
        
        int result = 0;
        if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") {
            result = visionmatrixsdk::falldetection::processImage(model, input_file.c_str(), output_file.c_str());
        } else if (extension == ".mp4" || extension == ".avi" || extension == ".mov" || 
                   extension == ".mkv" || extension == ".wmv") {
            result = visionmatrixsdk::falldetection::processVideo(model, input_file.c_str(), output_file.c_str());
        } else {
            std::cerr << "Error: Unsupported file format: " << extension << std::endl;
            std::cerr << "Supported formats: .jpg, .png, .mp4, .avi, .mov, .mkv, .wmv" << std::endl;
            result = 1;
        }
        
        // Cleanup
        visionmatrixsdk::falldetection::deinit(model);
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
