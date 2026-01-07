#pragma once

#include <vector>
#include <iostream>


struct DataPoint {
    std::vector<double> features;
    int label;
};

std::vector<DataPoint> read_csv(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    std::string line;
    getline(file, line);  // Skip header
    
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        
        while (getline(ss, value, ',')) {
            features.push_back(std::stod(value));
        }
        
        int label = features.back();
        features.pop_back();
        data.push_back({features, label});
    }
    return data;
}
