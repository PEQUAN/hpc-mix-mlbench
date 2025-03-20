#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>

struct DataPoint {
    std::vector<double> features;
    double target;
};

struct DecisionTreeRegressor {
    struct Node {
        bool is_leaf = false;
        double value = 0.0;
        double split_value = 0.0;
        int feature_index = -1;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };
    
    std::unique_ptr<Node> root;
    int max_depth;
    
    double calculate_variance(const std::vector<DataPoint>& data) {
        if (data.empty()) return 0.0;
        double mean = 0.0;
        for (const auto& point : data) mean += point.target;
        mean /= data.size();
        
        double variance = 0.0;
        for (const auto& point : data) {
            double diff = point.target - mean;
            variance += diff * diff;
        }
        return variance / data.size();
    }
    
    std::pair<int, double> find_best_split(const std::vector<DataPoint>& data, 
                                         const std::vector<int>& feature_indices) {
        if (data.empty() || feature_indices.empty()) return {-1, 0.0};
        
        double best_reduction = -std::numeric_limits<double>::infinity();
        int best_feature = -1;
        double best_value = 0.0;
        
        double total_variance = calculate_variance(data);
        
        for (int f : feature_indices) {
            std::vector<double> values;
            for (const auto& point : data) {
                if (f >= static_cast<int>(point.features.size())) return {-1, 0.0};
                values.push_back(point.features[f]);
            }
            std::sort(values.begin(), values.end());
            if (values.size() < 2) continue;
            
            for (size_t i = 0; i < values.size() - 1; ++i) {
                double split_val = (values[i] + values[i + 1]) / 2;
                std::vector<DataPoint> left, right;
                for (const auto& point : data) {
                    if (point.features[f] < split_val) left.push_back(point);
                    else right.push_back(point);
                }
                if (left.empty() || right.empty()) continue;
                
                double left_var = calculate_variance(left);
                double right_var = calculate_variance(right);
                double reduction = total_variance - 
                                 (left.size() * left_var + right.size() * right_var) / data.size();
                
                if (reduction > best_reduction) {
                    best_reduction = reduction;
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
            node->is_leaf = true;
            node->value = 0.0;
            return node;
        }
        
        if (depth >= max_depth || data.size() < 2) {
            node->is_leaf = true;
            double sum = 0.0;
            for (const auto& point : data) sum += point.target;
            node->value = sum / data.size();
            return node;
        }
        
        auto [feature, value] = find_best_split(data, feature_indices);
        if (feature == -1) {
            node->is_leaf = true;
            double sum = 0.0;
            for (const auto& point : data) sum += point.target;
            node->value = sum / data.size();
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
    DecisionTreeRegressor(int max_d = 15) : max_depth(max_d) {}  // Increased max_depth
    
    void fit(const std::vector<DataPoint>& data, const std::vector<int>& feature_indices) {
        if (data.empty()) {
            std::cerr << "Error: Empty dataset in DecisionTreeRegressor::fit" << std::endl;
            return;
        }
        root = build_tree(data, feature_indices, 0);
    }
    
    double predict(const std::vector<double>& features) {
        if (!root) return 0.0;
        Node* current = root.get();
        while (!current->is_leaf) {
            if (current->feature_index >= static_cast<int>(features.size())) return 0.0;
            current = (features[current->feature_index] < current->split_value) ? 
                      current->left.get() : current->right.get();
            if (!current) return 0.0;
        }
        return current->value;
    }
};

class RandomForestRegressor {
private:
    std::vector<DecisionTreeRegressor> trees;
    int n_trees;
    int max_depth;
    int max_features;
    unsigned int seed;
    
    std::vector<DataPoint> bootstrap_sample(const std::vector<DataPoint>& data, 
                                          std::mt19937& gen) {
        if (data.empty()) return {};
        std::vector<DataPoint> sample;
        std::uniform_int_distribution<> dis(0, data.size() - 1);
        for (size_t i = 0; i < data.size(); ++i) {
            sample.push_back(data[dis(gen)]);
        }
        return sample;
    }
    
    std::vector<int> random_features(int n_features, std::mt19937& gen) {
        if (n_features <= 0) return {};
        std::vector<int> all_features(n_features);
        std::iota(all_features.begin(), all_features.end(), 0);
        std::shuffle(all_features.begin(), all_features.end(), gen);
        return std::vector<int>(all_features.begin(), 
                               all_features.begin() + std::min(max_features, n_features));
    }

public:
    RandomForestRegressor(int n_t = 200, int m_d = 15, int m_f = -1, unsigned int s = 42)
        : n_trees(n_t), max_depth(m_d), max_features(m_f), seed(s) {}  // Increased n_trees
    
    void fit(const std::vector<DataPoint>& data) {
        if (data.empty()) {
            std::cerr << "Error: Empty dataset in RandomForestRegressor::fit" << std::endl;
            return;
        }
        std::mt19937 gen(seed);
        int n_features = data[0].features.size();
        if (max_features <= 0) max_features = static_cast<int>(sqrt(n_features)) + 1;
        
        trees.clear();
        for (int i = 0; i < n_trees; ++i) {
            std::vector<DataPoint> sample = bootstrap_sample(data, gen);
            std::vector<int> feature_indices = random_features(n_features, gen);
            if (sample.empty() || feature_indices.empty()) continue;
            DecisionTreeRegressor tree(max_depth);
            tree.fit(sample, feature_indices);
            trees.push_back(std::move(tree));
        }
    }
    
    double predict(const std::vector<double>& features) {
        if (trees.empty()) return 0.0;
        double sum = 0.0;
        int valid_trees = 0;
        for (auto& tree : trees) {
            double pred = tree.predict(features);
            sum += pred;
            valid_trees++;
        }
        return valid_trees > 0 ? sum / valid_trees : 0.0;
    }
};

std::vector<DataPoint> scale_features(const std::vector<DataPoint>& data) {
    if (data.empty()) return {};
    std::vector<DataPoint> scaled_data = data;
    int n_features = data[0].features.size();
    std::vector<double> means(n_features, 0.0);
    std::vector<double> stds(n_features, 0.0);
    
    for (const auto& point : data) {
        if (point.features.size() != n_features) return {};
        for (int i = 0; i < n_features; ++i) {
            means[i] += point.features[i];
        }
    }
    for (int i = 0; i < n_features; ++i) {
        means[i] /= data.size();
    }
    
    for (const auto& point : data) {
        for (int i = 0; i < n_features; ++i) {
            double diff = point.features[i] - means[i];
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
        std::cerr << "Error: Could not open " << filename << std::endl;
        return data;
    }
    std::string line;
    if (!getline(file, line)) {
        std::cerr << "Error: Empty file" << std::endl;
        return data;
    }
    std::cout << "Header: " << line << std::endl;
    
    int line_num = 1;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        double target = 0.0;
        int column = 0;
        
        while (getline(ss, value, ',')) {
            if (column == 0) {  // Skip index
                column++;
                continue;
            }
            try {
                if (column < 11) {  // Features (age to s6, columns 1-10)
                    features.push_back(std::stod(value));
                } else if (column == 11) {  // Target (label, column 11)
                    target = std::stod(value);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error parsing '" << value << "' at line " << line_num 
                          << ", column " << column << std::endl;
                return {};
            }
            column++;
        }
        
        if (features.size() != 10) {  // Expect exactly 10 features
            std::cerr << "Error: Expected 10 features, got " << features.size() 
                      << " at line " << line_num << ": " << line << std::endl;
            return {};
        }
        data.push_back({features, target});
        line_num++;
    }
    std::cout << "Loaded " << data.size() << " data points with " 
              << (data.empty() ? 0 : data[0].features.size()) << " features each" << std::endl;
    return data;
}

void write_predictions(const std::vector<DataPoint>& data, 
                      const std::vector<double>& predictions, 
                      const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing" << std::endl;
        return;
    }
    file << "age,sex,bmi,bp,s1,s2,s3,s4,s5,s6,target,prediction\n";
    
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].features.size(); ++j) {
            file << data[i].features[j];
            if (j < data[i].features.size() - 1) file << ",";
        }
        file << "," << data[i].target << "," << predictions[i] << "\n";
    }
}

int main() {
    std::vector<DataPoint> raw_data = read_csv("../data/regression/diabetes.csv");
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
    
    unsigned int random_seed = 42;
    RandomForestRegressor rf(30, 5, 10, random_seed);  // 30 trees, depth 5
    auto start = std::chrono::high_resolution_clock::now();
    rf.fit(train_data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    std::vector<double> predictions;
    double mse = 0.0;
    for (const auto& point : test_data) {
        double pred = rf.predict(point.features);
        predictions.push_back(pred);
        double diff = pred - point.target;
        mse += diff * diff;
    }
    mse /= test_data.size();
    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
    
    write_predictions(test_data, predictions, "results/randomforest/pred_reg.csv");
    
    return 0;
}
