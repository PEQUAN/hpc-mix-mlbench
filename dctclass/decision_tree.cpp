#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <cmath>

// Assuming same Node structure as before
struct Node {
    bool is_leaf = false;
    int class_label = -1;
    double split_value = 0.0;
    int feature_index = -1;
    Node* left = nullptr;
    Node* right = nullptr;
    ~Node() { delete left; delete right; }
};

struct DataPoint {
    std::vector<double> features;
    int label;
};

class DecisionTree {
private:
    Node* root;
    int max_depth;
    int min_samples_split;

    double calculate_entropy(const std::vector<DataPoint>& data) {
        std::map<int, int> label_counts;
        for (const auto& point : data) {
            label_counts[point.label]++;
        }
        double entropy = 0.0;
        for (const auto& pair : label_counts) {
            double p = static_cast<double>(pair.second) / data.size();
            entropy -= p * log2(p);
        }
        return entropy;
    }

    std::pair<int, double> find_best_split(const std::vector<DataPoint>& data) {
        double best_gain = -1.0;
        int best_feature = -1;
        double best_value = 0.0;
        double current_entropy = calculate_entropy(data);
        
        for (size_t f = 0; f < data[0].features.size(); ++f) {
            std::vector<double> values;
            for (const auto& point : data) {
                values.push_back(point.features[f]);
            }
            std::sort(values.begin(), values.end());
            
            for (size_t i = 0; i < values.size() - 1; ++i) {
                double split_val = (values[i] + values[i + 1]) / 2;
                std::vector<DataPoint> left, right;
                for (const auto& point : data) {
                    if (point.features[f] < split_val) {
                        left.push_back(point);
                    } else {
                        right.push_back(point);
                    }
                }
                if (left.empty() || right.empty()) continue;
                
                double gain = current_entropy - 
                    (static_cast<double>(left.size()) / data.size() * calculate_entropy(left) +
                     static_cast<double>(right.size()) / data.size() * calculate_entropy(right));
                
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = f;
                    best_value = split_val;
                }
            }
        }
        return {best_feature, best_value};
    }

    Node* build_tree(const std::vector<DataPoint>& data, int depth) {
        Node* node = new Node();
        if (depth >= max_depth || data.size() < min_samples_split) {
            node->is_leaf = true;
            std::map<int, int> counts;
            for (const auto& point : data) {
                counts[point.label]++;
            }
            node->class_label = std::max_element(counts.begin(), counts.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; })->first;
            return node;
        }
        
        auto [feature, value] = find_best_split(data);
        if (feature == -1) {
            node->is_leaf = true;
            std::map<int, int> counts;
            for (const auto& point : data) {
                counts[point.label]++;
            }
            node->class_label = std::max_element(counts.begin(), counts.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; })->first;
            return node;
        }
        
        std::vector<DataPoint> left_data, right_data;
        for (const auto& point : data) {
            if (point.features[feature] < value) {
                left_data.push_back(point);
            } else {
                right_data.push_back(point);
            }
        }
        
        node->feature_index = feature;
        node->split_value = value;
        node->left = build_tree(left_data, depth + 1);
        node->right = build_tree(right_data, depth + 1);
        return node;
    }

public:
    DecisionTree(int max_d = 10, int min_split = 2) 
        : max_depth(max_d), min_samples_split(min_split), root(nullptr) {}
    ~DecisionTree() { delete root; }
    
    void fit(const std::vector<DataPoint>& data) {
        root = build_tree(data, 0);
    }
    
    int predict(const std::vector<double>& features) {
        Node* current = root;
        while (!current->is_leaf) {
            if (features[current->feature_index] < current->split_value) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        return current->class_label;
    }
};

// Function to read CSV file
std::vector<DataPoint> read_csv(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    std::string line;
    
    // Skip header if it exists
    getline(file, line);
    
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        
        // Read features
        while (getline(ss, value, ',')) {
            features.push_back(std::stod(value));
        }
        
        // Last value is the label
        int label = features.back();
        features.pop_back();
        
        data.push_back({features, label});
    }
    return data;
}

// Function to write predictions to CSV
void write_predictions(const std::vector<DataPoint>& data, 
                      const std::vector<int>& predictions, 
                      const std::string& filename) {
    std::ofstream file(filename);
    file << "feature1,feature2,...,label,prediction\n";
    
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].features.size(); ++j) {
            file << data[i].features[j];
            if (j < data[i].features.size() - 1) file << ",";
        }
        file << "," << data[i].label << "," << predictions[i] << "\n";
    }
}

int main() {
    // Read data from CSV (assuming format: feature1,feature2,...,label)
    std::vector<DataPoint> data = read_csv("../data/iris/iris.csv");
    
    // Split into train and test (simple 80-20 split)
    size_t train_size = static_cast<size_t>(0.8 * data.size());
    std::vector<DataPoint> train_data(data.begin(), data.begin() + train_size);
    std::vector<DataPoint> test_data(data.begin() + train_size, data.end());
    
    // Train the model and measure time
    DecisionTree tree(3, 2);
    auto start = std::chrono::high_resolution_clock::now();
    tree.fit(train_data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    // Make predictions and calculate accuracy
    std::vector<int> predictions;
    int correct = 0;

    for (const auto& point : test_data) {
        int pred = tree.predict(point.features);
        predictions.push_back(pred);
        if (pred == point.label) correct++;
    }
    
    double accuracy = static_cast<double>(correct) / test_data.size() * 100;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    
    // Write predictions to CSV
    write_predictions(test_data, predictions, "../dct/predictions.csv");
    
    return 0;
}