#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>

const int MAX_SAMPLES = 1000;  // Maximum number of samples
const int N_FEATURES = 13;     // Number of features in Boston Housing dataset

struct DataPoint {
    double features[N_FEATURES];
    double target;
};

struct DecisionStump {
    int feature_index;
    double split_value;
    double left_value;
    double right_value;

    double predict(const double features[]) {
        return features[feature_index] < split_value ? left_value : right_value;
    }

    void fit(const DataPoint data[], const double weights[], int n_samples) {
        double best_error = std::numeric_limits<double>::infinity();
        
        for (int f = 0; f < N_FEATURES; ++f) {
            double values[MAX_SAMPLES];
            for (int i = 0; i < n_samples; ++i) values[i] = data[i].features[f];
            std::sort(values, values + n_samples);
            
            for (int i = 0; i < n_samples - 1; ++i) {
                double split = (values[i] + values[i + 1]) / 2;
                double left_sum = 0.0, right_sum = 0.0;
                double left_weight = 0.0, right_weight = 0.0;
                
                for (int j = 0; j < n_samples; ++j) {
                    if (data[j].features[f] < split) {
                        left_sum += weights[j] * data[j].target;
                        left_weight += weights[j];
                    } else {
                        right_sum += weights[j] * data[j].target;
                        right_weight += weights[j];
                    }
                }
                
                double left_val = left_weight > 0 ? left_sum / left_weight : 0.0;
                double right_val = right_weight > 0 ? right_sum / right_weight : 0.0;
                double error = 0.0;
                
                for (int j = 0; j < n_samples; ++j) {
                    double pred = data[j].features[f] < split ? left_val : right_val;
                    error += weights[j] * std::abs(data[j].target - pred);
                }
                
                if (error < best_error) {
                    best_error = error;
                    feature_index = f;
                    split_value = split;
                    left_value = left_val;
                    right_value = right_val;
                }
            }
        }
    }
};

class AdaBoostRegressor {
private:
    DecisionStump stumps[50];  // Fixed number of estimators
    double stump_weights[50];
    int n_estimators;
    int n_stumps;

public:
    AdaBoostRegressor(int n_est = 50) : n_estimators(n_est), n_stumps(0) {}
    
    void fit(const DataPoint data[], int n_samples) {
        if (n_samples == 0) return;
        double weights[MAX_SAMPLES];
        for (int i = 0; i < n_samples; ++i) weights[i] = 1.0 / n_samples;
        double total_weight = 1.0;

        for (int t = 0; t < n_estimators && n_stumps < 50; ++t) {
            DecisionStump stump;
            stump.fit(data, weights, n_samples);
            double error = 0.0;
            double residuals[MAX_SAMPLES];
            
            for (int i = 0; i < n_samples; ++i) {
                residuals[i] = std::abs(data[i].target - stump.predict(data[i].features));
                error += weights[i] * residuals[i];
            }
            error /= total_weight;
            if (error >= 0.5) break;
            
            double alpha = 0.5 * std::log((1.0 - error) / (error + 1e-10));
            stumps[n_stumps] = stump;
            stump_weights[n_stumps] = alpha;
            n_stumps++;
            
            total_weight = 0.0;
            for (int i = 0; i < n_samples; ++i) {
                weights[i] *= std::exp(alpha * residuals[i]);
                total_weight += weights[i];
            }
            for (int i = 0; i < n_samples; ++i) {
                weights[i] /= total_weight;
            }
        }
    }
    
    double predict(const double features[]) {
        double sum = 0.0;
        double weight_sum = 0.0;
        for (int i = 0; i < n_stumps; ++i) {
            sum += stump_weights[i] * stumps[i].predict(features);
            weight_sum += stump_weights[i];
        }
        return sum / (weight_sum + 1e-10);
    }
};

void scale_features(const DataPoint data[], DataPoint scaled_data[], int n_samples) {
    if (n_samples == 0) return;
    double means[N_FEATURES] = {0.0};
    double stds[N_FEATURES] = {0.0};
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < N_FEATURES; ++j) {
            means[j] += data[i].features[j];
        }
    }
    for (int j = 0; j < N_FEATURES; ++j) {
        means[j] /= n_samples;
    }
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < N_FEATURES; ++j) {
            double diff = data[i].features[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    for (int j = 0; j < N_FEATURES; ++j) {
        stds[j] = sqrt(stds[j] / n_samples);
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }
    
    for (int i = 0; i < n_samples; ++i) {
        scaled_data[i] = data[i];
        for (int j = 0; j < N_FEATURES; ++j) {
            scaled_data[i].features[j] = (data[i].features[j] - means[j]) / stds[j];
        }
    }
}

int read_csv(const std::string& filename, DataPoint data[], int& n_samples) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return 1;
    }
    std::string line;
    if (!getline(file, line)) return 1;
    std::cout << "Header: " << line << std::endl;
    
    n_samples = 0;
    int line_num = 1;
    while (getline(file, line) && n_samples < MAX_SAMPLES) {
        std::stringstream ss(line);
        std::string value;
        double features[N_FEATURES];
        double target = 0.0;
        int column = 0;
        
        while (getline(ss, value, ',') && column < N_FEATURES + 1) {
            try {
                if (value == "NA") value = "0"; // Handle NA values
                if (column < N_FEATURES) {
                    features[column] = std::stod(value);
                } else {
                    target = std::stod(value);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error parsing '" << value << "' at line " << line_num 
                          << ", column " << column << std::endl;
                return 1;
            }
            column++;
        }
        
        if (column != N_FEATURES + 1) {
            std::cerr << "Error: Expected " << N_FEATURES << " features + 1 target, got " 
                      << column << " at line " << line_num << std::endl;
            return 1;
        }
        for (int i = 0; i < N_FEATURES; ++i) {
            data[n_samples].features[i] = features[i];
        }
        data[n_samples].target = target;
        n_samples++;
        line_num++;
    }
    std::cout << "Loaded " << n_samples << " data points with " 
              << N_FEATURES << " features each" << std::endl;
    return 0;
}

void write_predictions(const DataPoint data[], const double predictions[], 
                     int n_samples, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    file << "CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV,prediction\n";
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < N_FEATURES; ++j) {
            file << data[i].features[j] << (j < N_FEATURES - 1 ? "," : "");
        }
        file << "," << data[i].target << "," << predictions[i] << "\n";
    }
}

int main() {
    DataPoint raw_data[MAX_SAMPLES];
    int n_samples = 0;
    
    if (read_csv("../data/regression/boston_housing.csv", raw_data, n_samples)) return 1;
    if (n_samples == 0) return 1;
    
    DataPoint scaled_data[MAX_SAMPLES];
    scale_features(raw_data, scaled_data, n_samples);
    
    int train_size = static_cast<int>(0.8 * n_samples);
    AdaBoostRegressor ada(50);
    auto start = std::chrono::high_resolution_clock::now();
    ada.fit(scaled_data, train_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    double predictions[MAX_SAMPLES];
    double mse = 0.0;
    for (int i = train_size; i < n_samples; ++i) {
        predictions[i - train_size] = ada.predict(scaled_data[i].features);
        double diff = predictions[i - train_size] - scaled_data[i].target;
        mse += diff * diff;
    }
    mse /= (n_samples - train_size);
    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
    
    write_predictions(&scaled_data[train_size], predictions, 
                     n_samples - train_size, "results/adaboost/preds_boston.csv");
    
    return 0;
}