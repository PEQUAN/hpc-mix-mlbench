#include <iostream>
#include <fstream>
#include <sstream>

#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "read.hpp"

class MLPClassifier {
private:
    int input_size;
    int hidden_size;
    int output_size;
    __PROMISE__ learning_rate;
    unsigned int seed;  
    std::vector<std::vector<__PROMISE__>> w1;
    std::vector<__PROMISE__> b1;
    std::vector<std::vector<__PROMISE__>> w2;
    std::vector<__PROMISE__> b2;
    
    __PROMISE__ sigmoid(__PROMISE__ x) {
        return 1.0 / (1.0 + exp(-x));
    }
    
    __PROMISE__ sigmoid_derivative(__PROMISE__ x) {
        __PROMISE__ s = sigmoid(x);
        return s * (1 - s);
    }
    
    std::vector<__PROMISE__> forward(const std::vector<__PROMISE__>& x, std::vector<__PROMISE__>& h) {
        std::vector<__PROMISE__> h_input(hidden_size, 0.0);
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < input_size; ++j) {
                h_input[i] += x[j] * w1[i][j];
            }
            h[i] = sigmoid(h_input[i] + b1[i]);
        }
        
        std::vector<__PROMISE__> o(output_size, 0.0);
        for (int i = 0; i < output_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                o[i] += h[j] * w2[i][j];
            }
            o[i] = sigmoid(o[i] + b2[i]);
        }
        return o;
    }
    
    void initialize_weights() {
        std::mt19937 gen(seed);  // Use provided seed
        __PROMISE__ limit1 = sqrt(6.0 / (input_size + hidden_size));
        __PROMISE__ limit2 = sqrt(6.0 / (hidden_size + output_size));
        std::uniform_real_distribution<> dis1(-limit1, limit1);
        std::uniform_real_distribution<> dis2(-limit2, limit2);
        
        w1.resize(hidden_size, std::vector<__PROMISE__>(input_size));
        b1.resize(hidden_size, 0.0);
        w2.resize(output_size, std::vector<__PROMISE__>(hidden_size));
        b2.resize(output_size, 0.0);
        
        for (auto& row : w1) {
            for (auto& val : row) val = dis1(gen);
        }
        for (auto& row : w2) {
            for (auto& val : row) val = dis2(gen);
        }
    }

public:
    MLPClassifier(int in_size, int hid_size, int out_size, double lr = 0.1, unsigned int s = 42)
        : input_size(in_size), hidden_size(hid_size), output_size(out_size), 
          learning_rate(lr), seed(s) {  // Added seed parameter with default value
        initialize_weights();
    }
    
    void fit(const std::vector<DataPoint>& data, int epochs = 100) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            for (const auto& point : data) {
                std::vector<__PROMISE__> h(hidden_size);
                std::vector<__PROMISE__> h_input(hidden_size, 0.0);
                for (int i = 0; i < hidden_size; ++i) {
                    for (int j = 0; j < input_size; ++j) {
                        h_input[i] += point.features[j] * w1[i][j];
                    }
                    h_input[i] += b1[i];
                    h[i] = sigmoid(h_input[i]);
                }
                
                std::vector<__PROMISE__> o(output_size, 0.0);
                std::vector<__PROMISE__> o_input(output_size, 0.0);
                for (int i = 0; i < output_size; ++i) {
                    for (int j = 0; j < hidden_size; ++j) {
                        o_input[i] += h[j] * w2[i][j];
                    }
                    o_input[i] += b2[i];
                    o[i] = sigmoid(o_input[i]);
                }
                
                std::vector<__PROMISE__> target(output_size, 0.0);
                target[point.label] = 1.0;
                
                for (int i = 0; i < output_size; ++i) {
                    total_loss += (o[i] - target[i]) * (o[i] - target[i]);
                }
                
                std::vector<__PROMISE__> o_delta(output_size);
                for (int i = 0; i < output_size; ++i) {
                    o_delta[i] = (o[i] - target[i]) * sigmoid_derivative(o_input[i]);
                }
                
                std::vector<__PROMISE__> h_delta(hidden_size);
                for (int i = 0; i < hidden_size; ++i) {
                    __PROMISE__ error = 0.0;
                    for (int j = 0; j < output_size; ++j) {
                        error += o_delta[j] * w2[j][i];
                    }
                    h_delta[i] = error * sigmoid_derivative(h_input[i]);
                }
                
                for (int i = 0; i < output_size; ++i) {
                    for (int j = 0; j < hidden_size; ++j) {
                        w2[i][j] -= learning_rate * o_delta[i] * h[j];
                    }
                    b2[i] -= learning_rate * o_delta[i];
                }
                
                for (int i = 0; i < hidden_size; ++i) {
                    for (int j = 0; j < input_size; ++j) {
                        w1[i][j] -= learning_rate * h_delta[i] * point.features[j];
                    }
                    b1[i] -= learning_rate * h_delta[i];
                }
            }
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << " Loss: " << total_loss / data.size() << std::endl;
            }
        }
    }
    
    int predict(const std::vector<__PROMISE__>& features) {
        std::vector<__PROMISE__> h(hidden_size);
        std::vector<__PROMISE__> output = forward(features, h);

        // PROMISE_CHECK_ARRAY(output.data(), output_size);
        PROMISE_CHECK_VAR(output[1]);
        return std::distance(output.begin(), 
                           std::max_element(output.begin(), output.end()));
    }
};

std::vector<DataPoint> scale_features(const std::vector<DataPoint>& data) {
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
    // Read and scale data
    std::vector<DataPoint> raw_data = read_csv("iris.csv");
    std::vector<DataPoint> data = scale_features(raw_data);
    
    // Train-test split (80-20)
    size_t train_size = static_cast<size_t>(0.5 * data.size());
    std::vector<DataPoint> train_data(data.begin(), data.begin() + train_size);
    std::vector<DataPoint> test_data(data.begin() + train_size, data.end());
    
    // MLP parameters
    int input_size = data[0].features.size();
    int output_size = 0;
    for (const auto& point : data) {
        output_size = std::max(output_size, point.label + 1);
    }
    int hidden_size = (input_size + output_size);
    
    // Train with specific seed
    unsigned int random_seed = 12345;  // You can change this value
    MLPClassifier classifier(input_size, hidden_size, output_size, 0.01, random_seed);
    auto start = std::chrono::high_resolution_clock::now();
    classifier.fit(train_data, 200);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    // Evaluate
    std::vector<int> predictions;
    float correct = 0.f;
    for (const auto& point : test_data) {
        int pred = classifier.predict(point.features);
        predictions.push_back(pred);
        if (pred == point.label) correct++;
    }
    
    __PROMISE__ accuracy = static_cast<__PROMISE__>(correct) / test_data.size() * 100;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    
    // Write predictions
    // write_predictions(test_data, predictions, "../../results/mlp/pred.csv");
    
    return 0;
}