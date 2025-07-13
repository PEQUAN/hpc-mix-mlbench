#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <map>
#include <memory>  

struct DataPoint {
    std::vector<double> features;
    int label;
};

struct DecisionTree {
    struct Node {
        bool is_leaf = false;
        int class_label = -1;
        float split_value = 0.0;
        int feature_index = -1;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };
    
    std::unique_ptr<Node> root;
    int max_depth;
    
    float calculate_gini(const std::vector<DataPoint>& data) {
        if (data.empty()) {
            std::cerr << "Warning: Empty data in calculate_gini" << std::endl;
            return 0.0;
        }
        std::map<int, int> counts;
        for (const auto& point : data) counts[point.label]++;
        float gini = 1.0;
        for (const auto& pair : counts) {
            float p = static_cast<half_float::half>(pair.second) / data.size();
            gini -= p * p;
        }
        return gini;
    }
    
    std::pair<int, float> find_best_split(const std::vector<DataPoint>& data, 
                                         const std::vector<int>& feature_indices) {
        if (data.empty() || feature_indices.empty()) {
            std::cerr << "Warning: Empty data or features in find_best_split" << std::endl;
            return {-1, 0.0};
        }
        
        half_float::half best_gini = std::numeric_limits<half_float::half>::infinity();
        int best_feature = -1;
        float best_value = 0.0;
        
        for (int f : feature_indices) {
            std::vector<float> values;
            for (const auto& point : data) {
                if (f >= static_cast<int>(point.features.size())) {
                    std::cerr << "Error: Feature index " << f << " out of bounds" << std::endl;
                    return {-1, 0.0};
                }
                values.push_back(point.features[f]);
            }
            std::sort(values.begin(), values.end());
            if (values.size() < 2) continue;
            
            for (size_t i = 0; i < values.size() - 1; ++i) {
                float split_val = (values[i] + values[i + 1]) / 2;
                std::vector<DataPoint> left, right;
                for (const auto& point : data) {
                    if (point.features[f] < split_val) left.push_back(point);
                    else right.push_back(point);
                }
                if (left.empty() || right.empty()) continue;
                
                float gini = (left.size() * calculate_gini(left) + 
                             right.size() * calculate_gini(right)) / data.size();
                
                if (gini < best_gini) {
                    best_gini = gini;
                    best_feature = f;
                    best_value = split_val;
                }
            }
        }
        return {best_feature, best_value};
    }
    
    std::unique_ptr<Node> build_tree(const std::vector<DataPoint>& data, 
                                   const std::vector<int>& feature_indices, int depth) {
        auto node = std::make_unique<Node>();
        
        if (data.empty()) {
            std::cerr << "Error: Empty data in build_tree at depth " << depth << std::endl;
            node->is_leaf = true;
            node->class_label = 0;
            return node;
        }
        
        if (depth >= max_depth || data.size() < 2) {
            node->is_leaf = true;
            std::map<int, int> counts;
            for (const auto& point : data) counts[point.label]++;
            node->class_label = counts.empty() ? 0 : 
                std::max_element(counts.begin(), counts.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; })->first;
            return node;
        }
        
        auto [feature, value] = find_best_split(data, feature_indices);
        if (feature == -1) {
            node->is_leaf = true;
            std::map<int, int> counts;
            for (const auto& point : data) counts[point.label]++;
            node->class_label = counts.empty() ? 0 : 
                std::max_element(counts.begin(), counts.end(),
                    [](const auto& a, const auto& b) { return a.second < b.second; })->first;
            return node;
        }
        
        std::vector<DataPoint> left_data, right_data;
        for (const auto& point : data) {
            if (point.features[feature] < value) left_data.push_back(point);
            else right_data.push_back(point);
        }
        
        node->feature_index = feature;
        node->split_value = value;
        node->left = build_tree(left_data, feature_indices, depth + 1);
        node->right = build_tree(right_data, feature_indices, depth + 1);
        
        return node;
    }
    
public:
    DecisionTree(int max_d = 10) : max_depth(max_d) {}
    
    void fit(const std::vector<DataPoint>& data, const std::vector<int>& feature_indices) {
        if (data.empty()) {
            std::cerr << "Error: Empty dataset in DecisionTree::fit" << std::endl;
            return;
        }
        root = build_tree(data, feature_indices, 0);
    }
    
    int predict(const std::vector<double>& features) {
        if (!root) {
            std::cerr << "Error: Tree not initialized in predict" << std::endl;
            return -1;
        }
        Node* current = root.get();
        while (!current->is_leaf) {
            if (current->feature_index >= static_cast<int>(features.size())) {
                std::cerr << "Error: Feature index " << current->feature_index 
                          << " exceeds feature size " << features.size() << std::endl;
                return -1;
            }
            current = (features[current->feature_index] < current->split_value) ? 
                      current->left.get() : current->right.get();

            PROMISE_CHECK_VAR(current->split_value);
            if (!current) {
                std::cerr << "Error: Null node in predict" << std::endl;
                return -1;
            }
        }
        return current->class_label;
    }
};

class RandomForest {
private:
    std::vector<DecisionTree> trees;
    int n_trees;
    int max_depth;
    int max_features;
    unsigned int seed;
    
    std::vector<DataPoint> bootstrap_sample(const std::vector<DataPoint>& data, 
                                          std::mt19937& gen) {
        if (data.empty()) {
            std::cerr << "Warning: Empty data in bootstrap_sample" << std::endl;
            return {};
        }
        std::vector<DataPoint> sample;
        std::uniform_int_distribution<> dis(0, data.size() - 1);
        for (size_t i = 0; i < data.size(); ++i) {
            sample.push_back(data[dis(gen)]);
        }
        return sample;
    }
    
    std::vector<int> random_features(int n_features, std::mt19937& gen) {
        if (n_features <= 0) {
            std::cerr << "Warning: No features in random_features" << std::endl;
            return {};
        }
        std::vector<int> all_features(n_features);
        std::iota(all_features.begin(), all_features.end(), 0);
        std::shuffle(all_features.begin(), all_features.end(), gen);
        return std::vector<int>(all_features.begin(), 
                               all_features.begin() + std::min(max_features, n_features));
    }

public:
    RandomForest(int n_t = 100, int m_d = 10, int m_f = -1, unsigned int s = 42)
        : n_trees(n_t), max_depth(m_d), max_features(m_f), seed(s) {}
    
    void fit(const std::vector<DataPoint>& data) {
        if (data.empty()) {
            std::cerr << "Error: Empty dataset in RandomForest::fit" << std::endl;
            return;
        }
        std::mt19937 gen(seed);
        int n_features = data[0].features.size();
        if (max_features <= 0) max_features = static_cast<int>(sqrt(n_features)) + 1;
        
        trees.clear();
        for (int i = 0; i < n_trees; ++i) {
            std::vector<DataPoint> sample = bootstrap_sample(data, gen);
            std::vector<int> feature_indices = random_features(n_features, gen);
            if (sample.empty() || feature_indices.empty()) {
                std::cerr << "Warning: Skipping tree " << i << " due to empty sample or features" << std::endl;
                continue;
            }
            DecisionTree tree(max_depth);
            tree.fit(sample, feature_indices);
            trees.push_back(std::move(tree));
        }
        if (trees.empty()) std::cerr << "Error: No trees built in RandomForest::fit" << std::endl;
    }
    
    int predict(const std::vector<double>& features) {
        if (trees.empty()) {
            std::cerr << "Error: No trees in RandomForest::predict" << std::endl;
            return -1;
        }
        std::map<int, int> votes;
        for (auto& tree : trees) {
            int pred = tree.predict(features);
            if (pred != -1) votes[pred]++;
        }
        if (votes.empty()) {
            std::cerr << "Error: No valid votes in predict" << std::endl;
            return -1;
        }
        return std::max_element(votes.begin(), votes.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; })->first;
    }
};

std::vector<DataPoint> scale_features(const std::vector<DataPoint>& data) {
    if (data.empty()) {
        std::cerr << "Error: Empty data in scale_features" << std::endl;
        return {};
    }
    std::vector<DataPoint> scaled_data = data;
    int n_features = data[0].features.size();
    std::vector<float> means(n_features, 0.0);
    std::vector<float> stds(n_features, 0.0);
    
    for (const auto& point : data) {
        if (point.features.size() != n_features) {
            std::cerr << "Error: Inconsistent feature count in data" << std::endl;
            return {};
        }
        for (int i = 0; i < n_features; ++i) {
            means[i] += point.features[i];
        }
    }
    for (int i = 0; i < n_features; ++i) {
        means[i] /= data.size();
    }
    
    for (const auto& point : data) {
        for (int i = 0; i < n_features; ++i) {
            float diff = point.features[i] - means[i];
            stds[i] += diff * diff;
        }
    }
    for (int i = 0; i < n_features; ++i) {
        stds[i] = sqrt(stds[i] / data.size());
        if (stds[i] < 1e-9) stds[i] = 1e-9;
    }
    
    for (auto& point : scaled_data) {
        for (int i = 0; i < n_features; ++i) {
            point.features[i] = (point.features[i] - means[i]) / stds[i];
        }
    }
    return scaled_data;
}


std::vector<DataPoint> read_csv(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    getline(file, line);  // Skip header: ,feature1,feature2,...,label
    
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        
        // Skip the index column
        getline(ss, value, ',');  // Ignore the first value (index)
        
        // Read features
        while (getline(ss, value, ',')) {
            features.push_back(std::stod(value));
        }

        // Last value is the true label
        int true_label = (int)features.back();
        features.pop_back();
        data.push_back({features, true_label});
    }
    std::cout << "Loaded " << data.size() << " data points with "  << (data.empty() ? 0 : data[0].features.size()) << " features each" << std::endl;
    
    file.close();
    return data;
}


void write_predictions(const std::vector<DataPoint>& data, 
                      const std::vector<int>& predictions, 
                      const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing" << std::endl;
        return;
    }
    file << "sepal length (cm),sepal width (cm),petal length (cm),petal width (cm),label,prediction\n";
    
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].features.size(); ++j) {
            file << data[i].features[j];
            if (j < data[i].features.size() - 1) file << ",";
        }
        file << "," << data[i].label << "," << predictions[i] << "\n";
    }
}

int main() {
    std::vector<DataPoint> raw_data = read_csv("iris.csv");
    if (raw_data.empty()) {
        std::cerr << "Error: No valid data loaded from CSV" << std::endl;
        return 1;
    }
    
    std::vector<DataPoint> data = scale_features(raw_data);
    if (data.empty()) {
        std::cerr << "Error: Feature scaling failed" << std::endl;
        return 1;
    }
    
    size_t train_size = static_cast<size_t>(0.8 * data.size());
    if (train_size == 0) {
        std::cerr << "Error: Dataset too small for train-test split" << std::endl;
        return 1;
    }
    std::vector<DataPoint> train_data(data.begin(), data.begin() + train_size);
    std::vector<DataPoint> test_data(data.begin() + train_size, data.end());
    
    unsigned int random_seed = 12345;
    RandomForest rf(100, 10, -1, random_seed);
    auto start = std::chrono::high_resolution_clock::now();
    rf.fit(train_data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    std::vector<int> predictions;
    int correct = 0;
    for (const auto& point : test_data) {
        int pred = rf.predict(point.features);
        if (pred == -1) {
            std::cerr << "Warning: Prediction failed, using default 0" << std::endl;
            pred = 0;
        }
        predictions.push_back(pred);
        if (pred == point.label) correct++;
    }
    
    float accuracy = static_cast<half_float::half>(correct) / test_data.size() * 100;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    
    return 0;
}