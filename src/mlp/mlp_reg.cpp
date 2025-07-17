#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <cmath>
#include <memory>

struct DataPoint {
    double* features;
    double target;
};

struct Layer {
    double* weights;  
    double* biases;
    int input_size;
    int output_size;
};

void free_datapoint(DataPoint* point, int n_features);
void free_layer(Layer* layer);
DataPoint* scale_features(DataPoint* data, int n_samples, int n_features);
DataPoint* read_csv(const std::string& filename, int& n_samples, int& n_features);
void write_predictions(DataPoint* data, double* predictions, int n_samples, int n_features, const std::string& filename);

class MLPRegressor {
private:
    Layer* layers;
    int n_layers;
    int n_features;
    double learning_rate;
    unsigned int seed;

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }
    
    double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }
    
public:
    MLPRegressor(int* layer_sizes, int n_layers, double lr = 0.01, unsigned int seed_val = 42) 
        : n_layers(n_layers), learning_rate(lr), seed(seed_val) {
        layers = new Layer[n_layers];
        n_features = layer_sizes[0];
        
        for (int i = 0; i < n_layers; ++i) {
            layers[i].input_size = layer_sizes[i];
            layers[i].output_size = layer_sizes[i + 1];
            layers[i].weights = new double[layer_sizes[i] * layer_sizes[i + 1]];
            layers[i].biases = new double[layer_sizes[i + 1]];
        }
    }
    
    ~MLPRegressor() {
        for (int i = 0; i < n_layers; ++i) {
            free_layer(&layers[i]);
        }
        delete[] layers;
    }
    
    void fit(DataPoint* data, int n_samples) {
        if (n_samples == 0) return;
        
        std::mt19937 gen(seed);
        std::normal_distribution<> dist(0.0, 1.0 / std::sqrt(n_features));
        
        // Initialize weights and biases
        for (int l = 0; l < n_layers; ++l) {
            for (int i = 0; i < layers[l].input_size * layers[l].output_size; ++i) {
                layers[l].weights[i] = dist(gen);
            }
            for (int i = 0; i < layers[l].output_size; ++i) {
                layers[l].biases[i] = dist(gen);
            }
        }

        int max_iter = 1000;
        // Allocate arrays for all layers' outputs
        int max_layer_size = 0;
        for (int l = 0; l < n_layers; ++l) {
            if (layers[l].output_size > max_layer_size) {
                max_layer_size = layers[l].output_size;
            }
        }
        double* layer_outputs = new double[n_samples * max_layer_size];
        double* layer_inputs = new double[n_samples * max_layer_size];
        double* temp_outputs = new double[max_layer_size];
        
        for (int iter = 0; iter < max_iter; ++iter) {
            // Forward pass
            for (int i = 0; i < n_samples; ++i) {
                double* current_input = data[i].features;
                int input_size = n_features;
                
                for (int l = 0; l < n_layers; ++l) {
                    int output_size = layers[l].output_size;
                    
                    for (int j = 0; j < output_size; ++j) {
                        double sum = layers[l].biases[j];
                        for (int k = 0; k < input_size; ++k) {
                            sum += current_input[k] * layers[l].weights[j * input_size + k];
                        }
                        layer_inputs[i * max_layer_size + j] = sum;
                        temp_outputs[j] = (l < n_layers - 1) ? sigmoid(sum) : sum;
                    }
                    
                    // Copy temp_outputs to layer_outputs
                    for (int j = 0; j < output_size; ++j) {
                        layer_outputs[i * max_layer_size + j] = temp_outputs[j];
                    }
                    
                    // Prepare for next layer
                    if (l < n_layers - 1) {
                        current_input = temp_outputs;
                        input_size = output_size;
                    }
                }
            }
            
            // Backward pass
            double* deltas = new double[n_samples * max_layer_size];
            double total_error = 0.0;
            
            // Calculate output layer deltas
            for (int i = 0; i < n_samples; ++i) {
                deltas[i * max_layer_size] = layer_outputs[i * max_layer_size] - data[i].target;
                total_error += deltas[i * max_layer_size] * deltas[i * max_layer_size];
            }
            total_error /= n_samples;
            
            // Backpropagation
            for (int l = n_layers - 1; l >= 0; --l) {
                int input_size = layers[l].input_size;
                int output_size = layers[l].output_size;
                double* prev_layer_output = (l == 0) ? data[0].features : 
                    layer_outputs + (l-1) * max_layer_size;
                
                // Update weights and biases
                for (int j = 0; j < output_size; ++j) {
                    double bias_grad = 0.0;
                    for (int k = 0; k < input_size; ++k) {
                        double weight_grad = 0.0;
                        for (int i = 0; i < n_samples; ++i) {
                            double delta = deltas[i * max_layer_size + j];
                            if (l < n_layers - 1) {
                                delta *= sigmoid_derivative(layer_inputs[i * max_layer_size + j]);
                            }
                            weight_grad += delta * (l == 0 ? data[i].features[k] : 
                                                  layer_outputs[i * max_layer_size + k - max_layer_size]);
                            if (k == 0) bias_grad += delta;
                        }
                        layers[l].weights[j * input_size + k] -= 
                            learning_rate * weight_grad / n_samples;
                    }
                    layers[l].biases[j] -= learning_rate * bias_grad / n_samples;
                }
                
                // Calculate deltas for previous layer
                if (l > 0) {
                    double* new_deltas = new double[n_samples * max_layer_size];
                    for (int i = 0; i < n_samples; ++i) {
                        for (int k = 0; k < input_size; ++k) {
                            double sum = 0.0;
                            for (int j = 0; j < output_size; ++j) {
                                double delta = deltas[i * max_layer_size + j];
                                if (l < n_layers - 1) {
                                    delta *= sigmoid_derivative(layer_inputs[i * max_layer_size + j]);
                                }
                                sum += delta * layers[l].weights[j * input_size + k];
                            }
                            new_deltas[i * max_layer_size + k] = sum;
                        }
                    }
                    delete[] deltas;
                    deltas = new_deltas;
                }
            }
            
            delete[] deltas;
            if (total_error < 1e-5) break;
        }
        
        delete[] layer_outputs;
        delete[] layer_inputs;
        delete[] temp_outputs;
    }
    
    double predict(const double* features) {
        double* current_input = new double[n_features];
        for (int i = 0; i < n_features; ++i) {
            current_input[i] = features[i];
        }
        int input_size = n_features;
        
        for (int l = 0; l < n_layers; ++l) {
            int output_size = layers[l].output_size;
            double* next_input = new double[output_size];
            
            for (int j = 0; j < output_size; ++j) {
                double sum = layers[l].biases[j];
                for (int k = 0; k < input_size; ++k) {
                    sum += current_input[k] * layers[l].weights[j * input_size + k];
                }
                next_input[j] = (l < n_layers - 1) ? sigmoid(sum) : sum;
            }
            
            delete[] current_input;
            current_input = next_input;
            input_size = output_size;
        }
        
        double result = current_input[0];
        delete[] current_input;
        return result;
    }
};

void free_datapoint(DataPoint* point, int n_features) {
    if (point && point->features) {
        delete[] point->features;
        point->features = nullptr;
    }
}

void free_layer(Layer* layer) {
    if (layer) {
        if (layer->weights) {
            delete[] layer->weights;
            layer->weights = nullptr;
        }
        if (layer->biases) {
            delete[] layer->biases;
            layer->biases = nullptr;
        }
    }
}

DataPoint* scale_features(DataPoint* data, int n_samples, int n_features) {
    if (n_samples == 0 || !data) return nullptr;
    DataPoint* scaled_data = new DataPoint[n_samples];
    double* means = new double[n_features]();
    double* stds = new double[n_features]();
    
    for (int i = 0; i < n_samples; ++i) {
        scaled_data[i].features = new double[n_features];
        scaled_data[i].target = data[i].target;
        for (int j = 0; j < n_features; ++j) {
            means[j] += data[i].features[j];
        }
    }
    
    for (int j = 0; j < n_features; ++j) {
        means[j] /= n_samples;
    }
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            double diff = data[i].features[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    
    for (int j = 0; j < n_features; ++j) {
        stds[j] = sqrt(stds[j] / n_samples);
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            scaled_data[i].features[j] = (data[i].features[j] - means[j]) / stds[j];
        }
    }
    
    delete[] means;
    delete[] stds;
    return scaled_data;
}

DataPoint* read_csv(const std::string& filename, int& n_samples, int& n_features) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return nullptr;
    }
    
    std::string line;
    if (!getline(file, line)) return nullptr;
    std::cout << "Header: " << line << std::endl;
    
    DataPoint* data = nullptr;
    n_samples = 0;
    n_features = 10;
    
    std::vector<DataPoint> temp_data;
    int line_num = 1;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        double* features = new double[n_features];
        double target = 0.0;
        int column = 0;
        int feature_idx = 0;
        
        while (getline(ss, value, ',')) {
            if (column == 0) {
                column++;
                continue;
            }
            try {
                if (column < 11) {
                    features[feature_idx++] = std::stod(value);
                } else if (column == 11) {
                    target = std::stod(value);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error parsing '" << value << "' at line " << line_num 
                          << ", column " << column << std::endl;
                delete[] features;
                return nullptr;
            }
            column++;
        }
        
        if (feature_idx != n_features) {
            std::cerr << "Error: Expected " << n_features << " features, got " 
                      << feature_idx << " at line " << line_num << std::endl;
            delete[] features;
            return nullptr;
        }
        temp_data.push_back({features, target});
        line_num++;
    }
    
    n_samples = temp_data.size();
    if (n_samples > 0) {
        data = new DataPoint[n_samples];
        for (int i = 0; i < n_samples; ++i) {
            data[i].features = temp_data[i].features;
            data[i].target = temp_data[i].target;
        }
    }
    
    std::cout << "Loaded " << n_samples << " data points with " 
              << n_features << " features each" << std::endl;
    return data;
}

void write_predictions(DataPoint* data, double* predictions, int n_samples, int n_features, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) return;
    file << "age,sex,bmi,bp,s1,s2,s3,s4,s5,s6,target,prediction\n";
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            file << data[i].features[j] << (j < n_features - 1 ? "," : "");
        }
        file << "," << data[i].target << "," << predictions[i] << "\n";
    }
}

int main() {
    int n_samples, n_features;
    DataPoint* raw_data = read_csv("../data/regression/diabetes.csv", n_samples, n_features);
    if (!raw_data) return 1;
    
    DataPoint* data = scale_features(raw_data, n_samples, n_features);
    if (!data) {
        for (int i = 0; i < n_samples; ++i) free_datapoint(&raw_data[i], n_features);
        delete[] raw_data;
        return 1;
    }
    
    int train_size = static_cast<int>(0.9 * n_samples);
    DataPoint* train_data = data;
    DataPoint* test_data = data + train_size;
    int test_size = n_samples - train_size;
    
    int layer_sizes[] = {n_features, 10, 8, 1}; // Input layer, two hidden layers, output layer
    MLPRegressor mlp(layer_sizes, 3, 0.01, 12345);
    
    auto start = std::chrono::high_resolution_clock::now();
    mlp.fit(train_data, train_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    double* predictions = new double[test_size];
    double mse = 0.0;
    try {
        for (int i = 0; i < test_size; ++i) {
            predictions[i] = mlp.predict(test_data[i].features);
            double diff = predictions[i] - test_data[i].target;
            mse += diff * diff;
        }
        mse /= test_size;
        std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
        
        write_predictions(test_data, predictions, test_size, n_features, "../results/pred_mlp.csv");
    } catch (const std::exception& e) {
        std::cerr << "Prediction error: " << e.what() << std::endl;
        delete[] predictions;
        for (int i = 0; i < n_samples; ++i) free_datapoint(&data[i], n_features);
        delete[] data;
        for (int i = 0; i < n_samples; ++i) free_datapoint(&raw_data[i], n_features);
        delete[] raw_data;
        return 1;
    }
    
    delete[] predictions;
    for (int i = 0; i < n_samples; ++i) free_datapoint(&data[i], n_features);
    delete[] data;
    for (int i = 0; i < n_samples; ++i) free_datapoint(&raw_data[i], n_features);
    delete[] raw_data;
    
    return 0;
}