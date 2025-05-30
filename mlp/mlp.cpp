#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>


#define MAX_SAMPLES 150 // for iris
#define MAX_FEATURES 4
#define MAX_CLASSES 3

struct DataPoint {
    double features[MAX_FEATURES];
    int label;
    int feature_count; 
};

class MLPClassifier {
private:
    int input_size;
    int hidden_size;
    int output_size;
    double learning_rate;
    unsigned int seed;
    double** w1; // [hidden_size][input_size]
    double* b1;  // [hidden_size]
    double** w2; // [output_size][hidden_size]
    double* b2;  // [output_size]

    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1 - s);
    }

    void forward(const double* x, double* h, double* o) {
        double h_input[MAX_SAMPLES];
        for (int i = 0; i < hidden_size; ++i) {
            h_input[i] = 0.0;
            for (int j = 0; j < input_size; ++j) {
                h_input[i] += x[j] * w1[i][j];
            }
            h[i] = sigmoid(h_input[i] + b1[i]);
        }

        for (int i = 0; i < output_size; ++i) {
            o[i] = 0.0;
            for (int j = 0; j < hidden_size; ++j) {
                o[i] += h[j] * w2[i][j];
            }
            o[i] = sigmoid(o[i] + b2[i]);
        }
    }

    void initialize_weights() {
        std::mt19937 gen(seed);
        double limit1 = sqrt(6.0 / (input_size + hidden_size));
        double limit2 = sqrt(6.0 / (hidden_size + output_size));
        std::uniform_real_distribution<> dis1(-limit1, limit1);
        std::uniform_real_distribution<> dis2(-limit2, limit2);

        w1 = new double*[hidden_size];
        for (int i = 0; i < hidden_size; ++i) {
            w1[i] = new double[input_size];
            for (int j = 0; j < input_size; ++j) {
                w1[i][j] = dis1(gen);
            }
        }

        b1 = new double[hidden_size]();
        w2 = new double*[output_size];
        for (int i = 0; i < output_size; ++i) {
            w2[i] = new double[hidden_size];
            for (int j = 0; j < hidden_size; ++j) {
                w2[i][j] = dis2(gen);
            }
        }

        b2 = new double[output_size]();
    }

public:
    MLPClassifier(int in_size, int hid_size, int out_size, double lr = 0.1, unsigned int s = 42)
        : input_size(in_size), hidden_size(hid_size), output_size(out_size),
          learning_rate(lr), seed(s) {
        if (in_size > MAX_FEATURES || hid_size > MAX_SAMPLES || out_size > MAX_CLASSES) {
            throw std::runtime_error("Array sizes exceed maximum limits");
        }
        initialize_weights();
    }

    ~MLPClassifier() {
        for (int i = 0; i < hidden_size; ++i) delete[] w1[i];
        delete[] w1;
        delete[] b1;
        for (int i = 0; i < output_size; ++i) delete[] w2[i];
        delete[] w2;
        delete[] b2;
    }

    void fit(DataPoint* data, int data_size, int epochs = 100) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            for (int d = 0; d < data_size; ++d) {
                double h[MAX_SAMPLES];
                double h_input[MAX_SAMPLES];
                double o[MAX_CLASSES];
                double o_input[MAX_CLASSES];

                for (int i = 0; i < hidden_size; ++i) {
                    h_input[i] = 0.0;
                    for (int j = 0; j < input_size; ++j) {
                        h_input[i] += data[d].features[j] * w1[i][j];
                    }
                    h_input[i] += b1[i];
                    h[i] = sigmoid(h_input[i]);
                }

                for (int i = 0; i < output_size; ++i) {
                    o_input[i] = 0.0;
                    for (int j = 0; j < hidden_size; ++j) {
                        o_input[i] += h[j] * w2[i][j];
                    }
                    o_input[i] += b2[i];
                    o[i] = sigmoid(o_input[i]);
                }

                double target[MAX_CLASSES] = {0.0};
                target[data[d].label] = 1.0;

                for (int i = 0; i < output_size; ++i) {
                    total_loss += (o[i] - target[i]) * (o[i] - target[i]);
                }

                double o_delta[MAX_CLASSES];
                for (int i = 0; i < output_size; ++i) {
                    o_delta[i] = (o[i] - target[i]) * sigmoid_derivative(o_input[i]);
                }

                double h_delta[MAX_SAMPLES];
                for (int i = 0; i < hidden_size; ++i) {
                    double error = 0.0;
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
                        w1[i][j] -= learning_rate * h_delta[i] * data[d].features[j];
                    }
                    b1[i] -= learning_rate * h_delta[i];
                }
            }
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << " Loss: " << total_loss / data_size << std::endl;
            }
        }
    }

    int predict(const double* features) {
        double h[MAX_SAMPLES];
        double o[MAX_CLASSES];
        forward(features, h, o);
        int max_idx = 0;
        for (int i = 1; i < output_size; ++i) {
            if (o[i] > o[max_idx]) max_idx = i;
        }
        return max_idx;
    }
};

void scale_features(DataPoint* data, int data_size, int n_features) {
    double means[MAX_FEATURES] = {0.0};
    double stds[MAX_FEATURES] = {0.0};

    for (int i = 0; i < data_size; ++i) {
        for (int j = 0; j < n_features; ++j) {
            means[j] += data[i].features[j];
        }
    }
    for (int j = 0; j < n_features; ++j) {
        means[j] /= data_size;
    }

    for (int i = 0; i < data_size; ++i) {
        for (int j = 0; j < n_features; ++j) {
            double diff = data[i].features[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    for (int j = 0; j < n_features; ++j) {
        stds[j] = sqrt(stds[j] / data_size);
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }

    for (int i = 0; i < data_size; ++i) {
        for (int j = 0; j < n_features; ++j) {
            data[i].features[j] = (data[i].features[j] - means[j]) / stds[j];
        }
    }
}

int read_csv(const std::string& filename, DataPoint* data, int max_size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    std::string line;
    getline(file, line); // Skip header

    int count = 0;
    while (getline(file, line) && count < max_size) {
        std::stringstream ss(line);
        std::string value;
        int feature_idx = 0;

        while (getline(ss, value, ',') && feature_idx < MAX_FEATURES + 1) {
            data[count].features[feature_idx] = std::stod(value);
            feature_idx++;
        }
        data[count].feature_count = feature_idx - 1;
        data[count].label = static_cast<int>(data[count].features[feature_idx - 1]);
        count++;
    }
    return count;
}

void write_predictions(const DataPoint* data, int data_size, const int* predictions, int n_features, const std::string& filename) {
    std::ofstream file(filename);
    file << "feature1";
    for (int i = 1; i < n_features; ++i) {
        file << ",feature" << (i + 1);
    }
    file << ",label,prediction\n";

    for (int i = 0; i < data_size; ++i) {
        for (int j = 0; j < n_features; ++j) {
            file << data[i].features[j];
            if (j < n_features - 1) file << ",";
        }
        file << "," << data[i].label << "," << predictions[i] << "\n";
    }
}

int main() {
    DataPoint raw_data[MAX_SAMPLES];
    int raw_data_size = read_csv("../data/classification/iris.csv", raw_data, MAX_SAMPLES);

    int n_features = raw_data[0].feature_count;
    scale_features(raw_data, raw_data_size, n_features);

    int train_size = static_cast<int>(0.8 * raw_data_size);
    int test_size = raw_data_size - train_size;
    DataPoint* train_data = raw_data;
    DataPoint* test_data = raw_data + train_size;

    int input_size = n_features;
    int output_size = 0;
    for (int i = 0; i < raw_data_size; ++i) {
        output_size = std::max(output_size, raw_data[i].label + 1);
    }
    int hidden_size = input_size + output_size;

    unsigned int random_seed = 12345;
    MLPClassifier classifier(input_size, hidden_size, output_size, 0.01, random_seed);
    auto start = std::chrono::high_resolution_clock::now();
    classifier.fit(train_data, train_size, 200);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Training time: " << duration.count() << " ms" << std::endl;

    int predictions[MAX_SAMPLES];
    int correct = 0;
    for (int i = 0; i < test_size; ++i) {
        predictions[i] = classifier.predict(test_data[i].features);
        if (predictions[i] == test_data[i].label) correct++;
    }

    double accuracy = static_cast<double>(correct) / test_size * 100;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    write_predictions(test_data, test_size, predictions, n_features, "../resutls/mlp/pred.csv");

    return 0;
}