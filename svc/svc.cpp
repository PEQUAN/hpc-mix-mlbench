#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>

struct DataPoint {
    std::vector<double> features;
    int label;  // Should be -1 or 1 for binary SVM
};

class SVM {
private:
    std::vector<double> weights;
    double bias;
    double C;  // Regularization parameter
    double tol;  // Numerical tolerance
    int max_passes;  // Maximum number of passes without change
    
    double kernel(const std::vector<double>& x1, const std::vector<double>& x2) {
        // Linear kernel
        double result = 0.0;
        for (size_t i = 0; i < x1.size(); ++i) {
            result += x1[i] * x2[i];
        }
        return result;
    }
    
    double predict_raw(const std::vector<double>& x) {
        double result = bias;
        for (size_t i = 0; i < weights.size(); ++i) {
            result += weights[i] * x[i];
        }
        return result;
    }

public:
    SVM(double c = 1.0, double tolerance = 0.001, int max_iter = 100)
        : C(c), tol(tolerance), max_passes(max_iter) {}
    
    void fit(const std::vector<DataPoint>& data) {
        int n_samples = data.size();
        int n_features = data[0].features.size();
        
        weights.resize(n_features, 0.0);
        bias = 0.0;
        std::vector<double> alphas(n_samples, 0.0);
        std::vector<double> errors(n_samples, 0.0);
        
        int passes = 0;
        while (passes < max_passes) {
            int num_changed = 0;
            for (int i = 0; i < n_samples; ++i) {
                double Ei = predict_raw(data[i].features) - data[i].label;
                if ((data[i].label * Ei < -tol && alphas[i] < C) || 
                    (data[i].label * Ei > tol && alphas[i] > 0)) {
                    
                    // Select j != i randomly
                    int j = i;
                    while (j == i) {
                        j = rand() % n_samples;
                    }
                    
                    double Ej = predict_raw(data[j].features) - data[j].label;
                    
                    double old_alpha_i = alphas[i];
                    double old_alpha_j = alphas[j];
                    
                    // Compute L and H (bounds for alpha_j)
                    double L, H;
                    if (data[i].label != data[j].label) {
                        L = std::max(0.0, alphas[j] - alphas[i]);
                        H = std::min(C, C + alphas[j] - alphas[i]);
                    } else {
                        L = std::max(0.0, alphas[i] + alphas[j] - C);
                        H = std::min(C, alphas[i] + alphas[j]);
                    }
                    if (L == H) continue;
                    
                    // Compute eta
                    double eta = 2.0 * kernel(data[i].features, data[j].features) - 
                               kernel(data[i].features, data[i].features) - 
                               kernel(data[j].features, data[j].features);
                    if (eta >= 0) continue;
                    
                    // Update alpha_j
                    alphas[j] -= data[j].label * (Ei - Ej) / eta;
                    alphas[j] = std::max(L, std::min(H, alphas[j]));
                    
                    if (fabs(alphas[j] - old_alpha_j) < tol) continue;
                    
                    // Update alpha_i
                    alphas[i] += data[i].label * data[j].label * (old_alpha_j - alphas[j]);
                    
                    // Update weights and bias
                    double b1 = bias - Ei - 
                              data[i].label * (alphas[i] - old_alpha_i) * 
                              kernel(data[i].features, data[i].features) - 
                              data[j].label * (alphas[j] - old_alpha_j) * 
                              kernel(data[i].features, data[j].features);
                    double b2 = bias - Ej - 
                              data[i].label * (alphas[i] - old_alpha_i) * 
                              kernel(data[i].features, data[j].features) - 
                              data[j].label * (alphas[j] - old_alpha_j) * 
                              kernel(data[j].features, data[j].features);
                    
                    if (0 < alphas[i] && alphas[i] < C) bias = b1;
                    else if (0 < alphas[j] && alphas[j] < C) bias = b2;
                    else bias = (b1 + b2) / 2.0;
                    
                    // Update weights
                    for (int k = 0; k < n_features; ++k) {
                        weights[k] += data[i].label * (alphas[i] - old_alpha_i) * data[i].features[k] +
                                    data[j].label * (alphas[j] - old_alpha_j) * data[j].features[k];
                    }
                    
                    num_changed++;
                }
            }
            if (num_changed == 0) passes++;
            else passes = 0;
        }
    }
    
    int predict(const std::vector<double>& features) {
        return predict_raw(features) > 0 ? 1 : -1;
    }
};

// Feature scaling function (unchanged)
std::vector<DataPoint> scale_features(const std::vector<DataPoint>& data) {
    std::vector<DataPoint> scaled_data = data;
    int n_features = data[0].features.size();
    std::vector<double> means(n_features, 0.0);
    std::vector<double> stds(n_features, 0.0);
    
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
        // Convert labels to -1/1 for SVM
        data.push_back({features, label == 0 ? -1 : 1});
    }
    return data;
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
        // Convert -1/1 back to 0/1 for output
        file << "," << (data[i].label == 1 ? 1 : 0) << "," 
             << (predictions[i] == 1 ? 1 : 0) << "\n";
    }
}

int main() {
    // Seed random number generator for SVM
    srand(time(nullptr));
    
    // Read and scale data
    std::vector<DataPoint> raw_data = read_csv("../data/iris/iris.csv");
    std::vector<DataPoint> data = scale_features(raw_data);
    
    // Train-test split (80-20)
    size_t train_size = static_cast<size_t>(0.8 * data.size());
    std::vector<DataPoint> train_data(data.begin(), data.begin() + train_size);
    std::vector<DataPoint> test_data(data.begin() + train_size, data.end());
    
    // Train
    SVM classifier(1.0, 0.001, 100);
    auto start = std::chrono::high_resolution_clock::now();
    classifier.fit(train_data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    // Evaluate
    std::vector<int> predictions;
    int correct = 0;
    for (const auto& point : test_data) {
        int pred = classifier.predict(point.features);
        predictions.push_back(pred);
        if (pred == point.label) correct++;
    }
    
    double accuracy = static_cast<double>(correct) / test_data.size() * 100;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    
    // Write predictions
    write_predictions(test_data, predictions, "../svc/predictions.csv");
    
    return 0;
}