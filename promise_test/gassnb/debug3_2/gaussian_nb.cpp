#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <chrono>

struct DataPoint {
    float* features;
    int feature_count;
    int label;
    
    DataPoint() : features(nullptr), feature_count(0), label(0) {}
    
    ~DataPoint() {
        delete[] features;
    }
};

class GaussianNB {
private:
    std::map<int, float> class_priors;
    std::map<int, float*> class_means;
    std::map<int, float*> class_variances;
    int n_features;

    flx::floatx<4, 3> gaussian_pdf(flx::floatx<4, 3> x, flx::floatx<8, 7> mean, float variance) {
        flx::floatx<4, 3> exponent = -((x - mean) * (x - mean)) / (2 * variance);
        return (1.0 / sqrt(2 * M_PI * variance)) * exp(exponent);
    }

public:
    GaussianNB() : n_features(0) {}
    
    ~GaussianNB() {
        for (auto& mean : class_means) {
            delete[] mean.second;
        }
        for (auto& var : class_variances) {
            delete[] var.second;
        }
    }

    void fit(const DataPoint* data, int data_size) {
        if (data_size == 0) return;
        n_features = data[0].feature_count;
        
        std::map<int, float**> class_features;
        std::map<int, int> class_counts;
        
        // Group features by class
        for (int i = 0; i < data_size; ++i) {
            class_priors[data[i].label]++;
            class_counts[data[i].label]++;
        }
        
        // Initialize storage for class features
        for (auto& count : class_counts) {
            int label = count.first;
            float** features = new float*[count.second];
            class_features[label] = features;
            count.second = 0; // Reset for indexing
        }
        
        // Store features
        for (int i = 0; i < data_size; ++i) {
            int label = data[i].label;
            class_features[label][class_counts[label]] = data[i].features;
            class_counts[label]++;
        }
        
        // Calculate priors
        for (auto& prior : class_priors) {
            prior.second /= data_size;
        }
        
        // Calculate means and variances
        for (const auto& class_data : class_features) {
            int label = class_data.first;
            float** feature_vectors = class_data.second;
            int count = class_counts[label];
            
            float* means = new float[n_features]();
            float* variances = new float[n_features]();
            
            // Calculate means
            for (int i = 0; i < count; ++i) {
                for (int j = 0; j < n_features; ++j) {
                    means[j] += feature_vectors[i][j];
                }
            }
            for (int j = 0; j < n_features; ++j) {
                means[j] /= count;
            }
            
            // Calculate variances
            for (int i = 0; i < count; ++i) {
                for (int j = 0; j < n_features; ++j) {
                    flx::floatx<4, 3> diff = feature_vectors[i][j] - means[j];
                    variances[j] += diff * diff;
                }
            }
            for (int j = 0; j < n_features; ++j) {
                variances[j] /= count;
                if (variances[j] < 1e-9) variances[j] = 1e-9;
            }
            
            class_means[label] = means;
            class_variances[label] = variances;
        }
        
        // Clean up class_features
        for (auto& class_data : class_features) {
            delete[] class_data.second;
        }
    }
    
    int predict(const float* features) {
        flx::floatx<4, 3> max_log_prob = -99999.9;
        int best_class = -1;
        
        for (const auto& prior : class_priors) {
            int label = prior.first;
            flx::floatx<4, 3> log_prob = log(prior.second);
            
            for (int i = 0; i < n_features; ++i) {
                float prob = gaussian_pdf(features[i], 
                                        class_means[label][i], 
                                        class_variances[label][i]);
                log_prob += log(prob);
            }
            
            if (log_prob > max_log_prob) {
                max_log_prob = log_prob;
                best_class = label;
            }
        }
        return best_class;
    }
};

DataPoint* read_csv(const std::string& filename, int& data_size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        data_size = 0;
        return nullptr;
    }

    std::string line;
    getline(file, line);
    
    // Count lines to pre-allocate
    data_size = 0;
    while (getline(file, line)) data_size++;
    file.clear();
    file.seekg(0);
    getline(file, line);
    
    DataPoint* data = new DataPoint[data_size];
    int current_index = 0;
    
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        
        getline(ss, value, ',');
        
        // Count features
        int feature_count = 0;
        std::stringstream feature_ss(line);
        getline(feature_ss, value, ',');
        while (getline(feature_ss, value, ',')) feature_count++;
        feature_count--; // Last is label
        
        float* features = new float[feature_count];
        ss.clear();
        ss.str(line);
        getline(ss, value, ',');
        
        int i = 0;
        while (getline(ss, value, ',')) {
            if (i < feature_count) {
                features[i] = std::stod(value);
            } else {
                data[current_index].label = std::stoi(value);
            }
            i++;
        }
        
        data[current_index].features = features;
        data[current_index].feature_count = feature_count;
        current_index++;
    }
    
    std::cout << "Loaded " << data_size << " data points with " 
              << (data_size > 0 ? data[0].feature_count : 0) << " features each" << std::endl;
    
    file.close();
    return data;
}


int main() {
    int data_size;
    DataPoint* data = read_csv("iris.csv", data_size);
    if (data_size == 0) return 1;
    
    int train_size = static_cast<int>(0.8 * data_size);
    DataPoint* train_data = data;
    DataPoint* test_data = data + train_size;
    int test_size = data_size - train_size;
    
    GaussianNB classifier;
    auto start = std::chrono::high_resolution_clock::now();
    classifier.fit(train_data, train_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    int* predictions = new int[test_size];
    int correct = 0;
    for (int i = 0; i < test_size; ++i) {
        predictions[i] = classifier.predict(test_data[i].features);
        if (predictions[i] == test_data[i].label) correct++;
    }
    
    flx::floatx<8, 7> accuracy = static_cast<flx::floatx<8, 7>>(correct) / test_size * 100;
    
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    
    delete[] predictions;
    delete[] data;
    
    return 0;
}