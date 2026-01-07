#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <memory>

struct DataPoint {
    std::vector<__PROMISE__> features;
    __PROMISE__ target;
};

// Forward declarations
std::vector<DataPoint> scale_features(const std::vector<DataPoint>& data);
std::vector<DataPoint> read_csv(const std::string& filename);
void write_predictions(const std::vector<DataPoint>& data, 
                     const std::vector<__PROMISE__>& predictions, 
                     const std::string& filename);

class MLPRegressor {
private:
    std::vector<std::vector<__PROMISE__>> hidden_weights;
    std::vector<__PROMISE__> output_weights;
    std::vector<__PROMISE__> hidden_bias;
    __PROMISE__ output_bias;
    int n_features;
    int n_hidden;
    __PROMISE__ learning_rate;
    unsigned int seed;

    __PROMISE__ sigmoid(__PROMISE__ x) {
        return 1.0 / (1.0 + exp(-x));
    }
    
    __PROMISE__ sigmoid_derivative(__PROMISE__ x) {
        __PROMISE__ s = sigmoid(x);
        return s * (1 - s);
    }

public:
    MLPRegressor(int hidden = 10, __PROMISE__ lr = 0.01, unsigned int seed_val = 42) 
        : n_hidden(hidden), learning_rate(lr), seed(seed_val) {}
    
    void fit(const std::vector<DataPoint>& data) {
        if (data.empty()) return;
        n_features = data[0].features.size();
        
        std::mt19937 gen(seed);
        std::normal_distribution<> dist(0.0, 1.0 / std::sqrt(n_features));
        
        hidden_weights.resize(n_hidden, std::vector<__PROMISE__>(n_features));
        hidden_bias.resize(n_hidden);
        output_weights.resize(n_hidden);
        for (int i = 0; i < n_hidden; ++i) {
            for (int j = 0; j < n_features; ++j) {
                hidden_weights[i][j] = dist(gen);
            }
            hidden_bias[i] = dist(gen);
            output_weights[i] = dist(gen);
        }
        output_bias = dist(gen);

        int max_iter = 1000;
        for (int iter = 0; iter < max_iter; ++iter) {
            std::vector<__PROMISE__> hidden_outputs(data.size() * n_hidden, 0.0);
            std::vector<__PROMISE__> predictions(data.size(), 0.0);
            
            for (size_t i = 0; i < data.size(); ++i) {
                for (int h = 0; h < n_hidden; ++h) {
                    __PROMISE__ sum = hidden_bias[h];
                    for (int f = 0; f < n_features; ++f) {
                        sum += data[i].features[f] * hidden_weights[h][f];
                    }
                    hidden_outputs[i * n_hidden + h] = sigmoid(sum);
                }
                __PROMISE__ sum = output_bias;
                for (int h = 0; h < n_hidden; ++h) {
                    sum += hidden_outputs[i * n_hidden + h] * output_weights[h];
                }
                predictions[i] = sum;
            }
            
            std::vector<__PROMISE__> output_deltas(data.size());
            std::vector<__PROMISE__> hidden_deltas(data.size() * n_hidden);
            __PROMISE__ total_error = 0.0;
            
            for (size_t i = 0; i < data.size(); ++i) {
                output_deltas[i] = predictions[i] - data[i].target;
                total_error += output_deltas[i] * output_deltas[i];
                
                for (int h = 0; h < n_hidden; ++h) {
                    hidden_deltas[i * n_hidden + h] = output_deltas[i] * output_weights[h] * 
                        sigmoid_derivative(hidden_outputs[i * n_hidden + h]);
                }
            }
            total_error /= data.size();
            
            for (int h = 0; h < n_hidden; ++h) {
                for (int f = 0; f < n_features; ++f) {
                    __PROMISE__ grad = 0.0;
                    for (size_t i = 0; i < data.size(); ++i) {
                        grad += hidden_deltas[i * n_hidden + h] * data[i].features[f];
                    }
                    hidden_weights[h][f] -= learning_rate * grad / data.size();
                }
                __PROMISE__ bias_grad = 0.0;
                for (size_t i = 0; i < data.size(); ++i) {
                    bias_grad += hidden_deltas[i * n_hidden + h];
                }
                hidden_bias[h] -= learning_rate * bias_grad / data.size();
                
                __PROMISE__ weight_grad = 0.0;
                for (size_t i = 0; i < data.size(); ++i) {
                    weight_grad += output_deltas[i] * hidden_outputs[i * n_hidden + h];
                }
                output_weights[h] -= learning_rate * weight_grad / data.size();
            }
            __PROMISE__ output_bias_grad = 0.0;
            for (size_t i = 0; i < data.size(); ++i) {
                output_bias_grad += output_deltas[i];
            }
            output_bias -= learning_rate * output_bias_grad / data.size();
            
            if (total_error < 1e-5) break;
        }
    }
    
    __PROMISE__ predict(const std::vector<__PROMISE__>& features) {
        if (features.size() != static_cast<size_t>(n_features)) {
            throw std::runtime_error("Feature size mismatch in prediction");
        }
        std::vector<__PROMISE__> hidden(n_hidden);
        for (int h = 0; h < n_hidden; ++h) {
            __PROMISE__ sum = hidden_bias[h];
            for (int f = 0; f < n_features; ++f) {
                sum += features[f] * hidden_weights[h][f];
            }
            hidden[h] = sigmoid(sum);
        }

        __PROMISE__ sum = output_bias;
        for (int h = 0; h < n_hidden; ++h) {
            sum += hidden[h] * output_weights[h];
        }
        return sum;
    }
};

std::vector<DataPoint> scale_features(const std::vector<DataPoint>& data) {
    if (data.empty()) return {};
    std::vector<DataPoint> scaled_data = data;
    int n_features = data[0].features.size();
    std::vector<__PROMISE__> means(n_features, 0.0);
    std::vector<__PROMISE__> stds(n_features, 0.0);
    
    for (const auto& point : data) {
        for (int i = 0; i < n_features; ++i) {
            means[i] += point.features[i];
        }
    }
    for (int i = 0; i < n_features; ++i) {
        means[i] /= data.size();
    }
    
    for (const auto& point : data) {
        for (int i = 0; i < n_features; ++i) {
            __PROMISE__ diff = point.features[i] - means[i];
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
    if (!getline(file, line)) return data;
    std::cout << "Header: " << line << std::endl;
    
    int line_num = 1;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        double target = 0.0;
        int column = 0;
        
        while (getline(ss, value, ',')) {
            if (column == 0) {  // Skip index if present
                column++;
                continue;
            }
            try {
                if (column < 11) {  // Features (age to s6)
                    features.push_back(std::stod(value));
                } else if (column == 11) {  // Target (label)
                    target = std::stod(value);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error parsing '" << value << "' at line " << line_num 
                          << ", column " << column << std::endl;
                return {};
            }
            column++;
        }
        
        if (features.size() != 10) {
            std::cerr << "Error: Expected 10 features, got " << features.size() 
                      << " at line " << line_num << std::endl;
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
    if (!file.is_open()) return;
    file << "age,sex,bmi,bp,s1,s2,s3,s4,s5,s6,target,prediction\n";
    
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].features.size(); ++j) {
            file << data[i].features[j] << (j < data[i].features.size() - 1 ? "," : "");
        }
        file << "," << data[i].target << "," << predictions[i] << "\n";
    }
}

int main() {
    std::vector<DataPoint> raw_data = read_csv("diabetes.csv");
    if (raw_data.empty()) return 1;
    
    std::vector<DataPoint> data = scale_features(raw_data);
    if (data.empty()) return 1;
    
    size_t train_size = static_cast<size_t>(0.8 * data.size());
    std::vector<DataPoint> train_data(data.begin(), data.begin() + train_size);
    std::vector<DataPoint> test_data(data.begin() + train_size, data.end());
    
    MLPRegressor mlp(10, 0.01, 12345);
    auto start = std::chrono::high_resolution_clock::now();
    mlp.fit(train_data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    std::vector<__PROMISE__> predictions;
    __PROMISE__ mse = 0.0;
    try {
        for (const auto& point : test_data) {
            __PROMISE__ pred = mlp.predict(point.features);
            predictions.push_back(pred);
            __PROMISE__ diff = pred - point.target;
            mse += diff * diff;
        }
        mse /= test_data.size();
        std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
        
        PROMISE_CHECK_ARRAY(predictions.data(), predictions.size());
        // write_predictions(test_data, predictions, "../results/pred_mlp.csv");
    } catch (const std::exception& e) {
        std::cerr << "Prediction error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}