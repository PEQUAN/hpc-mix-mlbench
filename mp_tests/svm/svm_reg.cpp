#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>

const int MAX_SAMPLES = 1000; // Maximum number of samples
const int N_FEATURES = 4;    // CRIM, RM, LSTAT, PTRATIO
const double C = 1.0;        // Regularization parameter
const double EPSILON = 0.1;  // Epsilon-tube for regression

struct DataPoint {
    __PROMISE__ features[N_FEATURES];
    __PROMISE__ target; 
};

class SVMRegressor {
private:
    __PROMISE__ weights[N_FEATURES];
    __PROMISE__ bias;

    __PROMISE__ dot_product(const __PROMISE__* x1, const __PROMISE__* x2) {
        __PROMISE__ sum = 0.0;
        for (int i = 0; i < N_FEATURES; ++i) sum += x1[i] * x2[i];
        return sum;
    }

    __PROMISE__ predict_raw(const __PROMISE__* x) {
        return dot_product(weights, x) + bias;
    }

public:
    SVMRegressor() {
        for (int i = 0; i < N_FEATURES; ++i) weights[i] = 0.0;
        bias = 0.0;
    }
    
    void fit(const DataPoint* data, int n_data) {
        if (n_data == 0) return;

        // Simple gradient descent for linear SVR
        __PROMISE__ learning_rate = 0.001;
        int max_iter = 1000;
        for (int iter = 0; iter < max_iter; ++iter) {
            __PROMISE__ gradient_bias = 0.0;
            __PROMISE__ gradient_weights[N_FEATURES] = {0.0};
            __PROMISE__ total_error = 0.0;

            for (int j = 0; j < n_data; ++j) {
                __PROMISE__ pred = predict_raw(data[j].features);
                __PROMISE__ error = pred - data[j].target;

                if (abs(error) > EPSILON) {
                    __PROMISE__ grad = (error > EPSILON) ? 1.0 : -1.0;
                    gradient_bias += grad;
                    for (int i = 0; i < N_FEATURES; ++i) {
                        gradient_weights[i] += grad * data[j].features[i];
                    }
                    total_error += abs(error) - EPSILON;
                }
            }

            // Update weights and bias
            for (int i = 0; i < N_FEATURES; ++i) {
                weights[i] -= learning_rate * (gradient_weights[i] + 2 * C * weights[i]);
            }
            bias -= learning_rate * gradient_bias;

            if (total_error < 1e-5) break; // Early stopping
        }
    }
    
    __PROMISE__ predict(const __PROMISE__* features) {
        return predict_raw(features);
    }
};

void compute_feature_stats(const DataPoint data[], int n_samples, __PROMISE__ means[], __PROMISE__ stds[]) {
    int counts[N_FEATURES] = {0};
    for (int j = 0; j < N_FEATURES; ++j) {
        means[j] = 0.0;
        stds[j] = 0.0;
    }
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < N_FEATURES; ++j) {
            if (!isnan(data[i].features[j])) {
                means[j] += data[i].features[j];
                counts[j]++;
            }
        }
    }
    for (int j = 0; j < N_FEATURES; ++j) {
        double temp = 0.0;
        means[j] = counts[j] > temp ? means[j] / counts[j] : temp;
    }
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < N_FEATURES; ++j) {
            if (!isnan(data[i].features[j])) {
                __PROMISE__ diff = data[i].features[j] - means[j];
                stds[j] += diff * diff;
            }
        }
    }
    
    for (int j = 0; j < N_FEATURES; ++j) {
        double temp = 0.0;
        double temp2 =  1e-9;
        stds[j] = counts[j] > temp ? sqrt(stds[j] / counts[j]) :temp2;
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }
}

void transform_features(DataPoint data[], int n_samples) {
    // Apply log-transformation to CRIM (0) and LSTAT (2)
    int indices[] = {0, 2}; // CRIM, LSTAT in selected features
    for (int i = 0; i < n_samples; ++i) {
        for (int j : indices) {
            if (!isnan(data[i].features[j]) && data[i].features[j] > 0) {
                data[i].features[j] = log(data[i].features[j] + 1e-10);
            }
        }
    }
}

int remove_outliers(DataPoint data[], int n_samples, int& new_n_samples) {
    __PROMISE__ means[N_FEATURES];
    __PROMISE__ stds[N_FEATURES];
    compute_feature_stats(data, n_samples, means, stds);
    
    DataPoint temp_data[MAX_SAMPLES];
    new_n_samples = 0;
    for (int i = 0; i < n_samples; ++i) {
        bool is_outlier = false;
        for (int j = 0; j < N_FEATURES; ++j) {
            if (!isnan(data[i].features[j])) {
                __PROMISE__ z = abs((data[i].features[j] - means[j]) / stds[j]);
                if (z > 3.0) {
                    is_outlier = true;
                    break;
                }
            }
        }
        if (!is_outlier) {
            temp_data[new_n_samples] = data[i];
            new_n_samples++;
        }
    }
    
    for (int i = 0; i < new_n_samples; ++i) {
        data[i] = temp_data[i];
    }
    return new_n_samples > 0 ? 0 : 1;
}

void scale_features(const DataPoint data[], DataPoint scaled_data[], int n_samples) {
    if (n_samples == 0) return;
    __PROMISE__ means[N_FEATURES];
    __PROMISE__ stds[N_FEATURES];
    compute_feature_stats(data, n_samples, means, stds);
    
    for (int i = 0; i < n_samples; ++i) {
        scaled_data[i] = data[i];
        for (int j = 0; j < N_FEATURES; ++j) {
            if (!isnan(scaled_data[i].features[j])) {
                scaled_data[i].features[j] = (scaled_data[i].features[j] - means[j]) / stds[j];
            } else {
                scaled_data[i].features[j] = 0.0;
            }
        }
    }
}

void shuffle_data(DataPoint data[], int n_samples) {
    std::mt19937 gen(42); // Match random_state=42
    for (int i = n_samples - 1; i > 0; --i) {
        std::uniform_int_distribution<> dis(0, i);
        int j = dis(gen);
        DataPoint temp = data[i];
        data[i] = data[j];
        data[j] = temp;
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
    
    // Verify header
    std::stringstream header_ss(line);
    std::string value;
    int header_cols = 0;
    while (getline(header_ss, value, ',')) header_cols++;
    if (header_cols != 14) {
        std::cerr << "Error: Expected 14 columns in header, got " << header_cols << std::endl;
        return 1;
    }
    
    n_samples = 0;
    int line_num = 1;
    while (getline(file, line) && n_samples < MAX_SAMPLES) {
        std::stringstream ss(line);
        std::string value;
        __PROMISE__ features[14]; // Temporary for 13 features + 1 target
        bool valid_row = true;
        int column = 0;
        
        // Initialize features with NaN
        for (int i = 0; i < 14; ++i) {
            features[i] = std::numeric_limits<__PROMISE__>::quiet_NaN();
        }

        // Read all columns
        while (getline(ss, value, ',')) {
            if (column >= 14) break; // Prevent overflow
            if (value.empty() || value == "NA") {
                features[column] = std::numeric_limits<__PROMISE__>::quiet_NaN();
            } else {
                try {
                    features[column] = std::stod(value);
                } catch (const std::exception& e) {
                    std::cerr << "Error parsing '" << value << "' at line " << line_num 
                              << ", column " << column << std::endl;
                    return 1;
                }
            }
            column++;
        }
        
        if (column < 14) {
            std::cerr << "Warning: Expected 14 columns, got " << column 
                      << " at line " << line_num << ", imputing missing values" << std::endl;
            valid_row = false;
        }
        
        // Map to selected features: CRIM(0), RM(5), LSTAT(12), PTRATIO(10)
        int feature_map[] = {0, 5, 12, 10};
        for (int i = 0; i < N_FEATURES; ++i) {
            data[n_samples].features[i] = features[feature_map[i]];
        }
        data[n_samples].target = features[13]; // MEDV
        
        if (isnan(data[n_samples].target)) {
            std::cerr << "Warning: Missing target at line " << line_num << ", skipping" << std::endl;
            line_num++;
            continue;
        }
        
        n_samples++;
        line_num++;
    }
    
    // Impute missing values with feature means
    __PROMISE__ means[N_FEATURES] = {0.0};
    int counts[N_FEATURES] = {0};
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < N_FEATURES; ++j) {
            if (!isnan(data[i].features[j])) {
                means[j] += data[i].features[j];
                counts[j]++;
            }
        }
    }
    for (int j = 0; j < N_FEATURES; ++j) {
        double temp = 0.0;
        means[j] = counts[j] > 0 ? means[j] / counts[j] : temp;
    }
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < N_FEATURES; ++j) {
            if (isnan(data[i].features[j])) {
                data[i].features[j] = means[j];
            }
        }
    }
    
    std::cout << "Loaded " << n_samples << " data points with " 
              << N_FEATURES << " features each" << std::endl;
    return 0;
}


__PROMISE__ compute_r2_score(const DataPoint data[], const __PROMISE__ predictions[], int n_samples) {
    __PROMISE__ mean_y = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        mean_y += data[i].target;
    }
    mean_y /= n_samples;
    
    __PROMISE__ ss_tot = 0.0, ss_res = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        __PROMISE__ diff = data[i].target - mean_y;
        ss_tot += diff * diff;
        diff = data[i].target - predictions[i];
        ss_res += diff * diff;
    }
    return 1.0 - (ss_res / (ss_tot + 1e-10));
}

int main() {
    DataPoint raw_data[MAX_SAMPLES];
    int n_samples = 0;
    
    if (read_csv("boston_housing.csv", raw_data, n_samples)) return 1;
    if (n_samples == 0) return 1;
    
    // Apply log-transformation
    transform_features(raw_data, n_samples);
    
    // Remove outliers
    int new_n_samples = 0;
    if (remove_outliers(raw_data, n_samples, new_n_samples)) return 1;
    n_samples = new_n_samples;
    std::cout << "After outlier removal: " << n_samples << " data points" << std::endl;
    
    shuffle_data(raw_data, n_samples); // Shuffle data
    
    DataPoint scaled_data[MAX_SAMPLES]; // Scale features
    scale_features(raw_data, scaled_data, n_samples);
    
    int train_size = static_cast<int>(0.8 * n_samples);
    int test_size = n_samples - train_size;
    
    SVMRegressor svr;
    auto start = std::chrono::high_resolution_clock::now();
    svr.fit(scaled_data, train_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    
    __PROMISE__ predictions[MAX_SAMPLES];
    __PROMISE__ check_predictions[test_size];
    DataPoint* test_data = scaled_data + train_size;
    __PROMISE__ mse = 0.0;
    for (int i = 0; i < test_size; ++i) {
        predictions[i] = svr.predict(test_data[i].features);
        check_predictions[i] = predictions[i];
        __PROMISE__ diff = predictions[i] - test_data[i].target;
        mse += diff * diff;
    }
    mse /= test_size;
    PROMISE_CHECK_ARRAY(check_predictions, test_size);
    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
    
    __PROMISE__ r2 = compute_r2_score(test_data, predictions, test_size);
    std::cout << "R^2 Score: " << r2 << std::endl;
    
    return 0;
}