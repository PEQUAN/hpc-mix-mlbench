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
    int label;  // -1 or 1 for AdaBoost
};

struct DecisionStump {
    int feature_index;
    double threshold;
    double polarity;
    double alpha;
    
    int predict(const std::vector<double>& features) const {
        double value = features[feature_index];
        return (polarity * (value < threshold ? -1 : 1));
    }
};

class AdaBoostClassifier {
private:
    std::vector<DecisionStump> stumps;
    int n_estimators;
    unsigned int seed;

    DecisionStump train_stump(const std::vector<DataPoint>& data, 
                            const std::vector<double>& weights) {
        int n_features = data[0].features.size();
        DecisionStump best_stump{-1, 0.0, 1.0, 0.0};
        double min_error = std::numeric_limits<double>::infinity();
        
        for (int f = 0; f < n_features; ++f) {
            std::vector<double> values;
            for (const auto& point : data) {
                values.push_back(point.features[f]);
            }
            std::sort(values.begin(), values.end());
            
            for (size_t i = 0; i < values.size() - 1; ++i) {
                double threshold = (values[i] + values[i + 1]) / 2;
                for (double polarity : {1.0, -1.0}) {
                    double error = 0.0;
                    for (size_t j = 0; j < data.size(); ++j) {
                        int pred = polarity * (data[j].features[f] < threshold ? -1 : 1);
                        if (pred != data[j].label) error += weights[j];
                    }
                    if (error < min_error) {
                        min_error = error;
                        best_stump = {f, threshold, polarity, 0.0};
                    }
                }
            }
        }
        
        best_stump.alpha = 0.5 * log((1.0 - min_error) / (min_error + 1e-10));
        return best_stump;
    }

public:
    AdaBoostClassifier(int n_est = 50, unsigned int s = 42) 
        : n_estimators(n_est), seed(s) {}
    
    void fit(const std::vector<DataPoint>& data) {
        int n_samples = data.size();
        std::vector<double> weights(n_samples, 1.0 / n_samples);
        stumps.clear();
        
        for (int t = 0; t < n_estimators; ++t) {
            DecisionStump stump = train_stump(data, weights);
            stumps.push_back(stump);
            
            double total_weight = 0.0;
            for (int i = 0; i < n_samples; ++i) {
                int pred = stump.predict(data[i].features);
                weights[i] *= exp(-stump.alpha * data[i].label * pred);
                total_weight += weights[i];
            }
            
            for (auto& w : weights) {
                w /= total_weight;
            }
            
            if (stump.alpha < 0.01) break;
        }
    }
    
    int predict(const std::vector<double>& features) {
        double sum = 0.0;
        for (const auto& stump : stumps) {
            sum += stump.alpha * stump.predict(features);
        }
        return sum > 0 ? 1 : -1;
    }
};

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
    getline(file, line); 
    
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        
        while (getline(ss, value, ',')) {
            features.push_back(std::stod(value));
        }
        
        int label = features.back();
        features.pop_back();
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
        file << "," << (data[i].label == 1 ? 1 : 0) << "," 
             << (predictions[i] == 1 ? 1 : 0) << "\n";
    }
}

int main() {
    std::vector<DataPoint> raw_data = read_csv("../data/classification/iris.csv");
    std::vector<DataPoint> data = scale_features(raw_data);
    
    size_t train_size = static_cast<size_t>(0.8 * data.size());
    std::vector<DataPoint> train_data(data.begin(), data.begin() + train_size);
    std::vector<DataPoint> test_data(data.begin() + train_size, data.end());
    
    unsigned int random_seed = 12345;
    AdaBoostClassifier classifier(50, random_seed);
    auto start = std::chrono::high_resolution_clock::now();
    classifier.fit(train_data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    std::vector<int> predictions;
    int correct = 0;
    for (const auto& point : test_data) {
        int pred = classifier.predict(point.features);
        predictions.push_back(pred);
        if (pred == point.label) correct++;
    }
    
    double accuracy = static_cast<double>(correct) / test_data.size() * 100;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    
    write_predictions(test_data, predictions, "../results/adaboost/pred.csv");
    
    return 0;
}