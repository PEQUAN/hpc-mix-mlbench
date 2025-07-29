#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

const int MAX_SAMPLES = 1000;  // Maximum number of samples
const int N_FEATURES = 3;     // Number of features after dropping RAD , increase to 4, mse=3
const int MAX_ESTIMATORS = 2; // Number of estimators
const flx::floatx<5, 2> LEARNING_RATE = 1; // Learning rate for AdaBoost

struct DataPoint {
    double features[N_FEATURES];
    double target;  // MEDV
};

struct DecisionTree {
    int feature_index1;
    flx::floatx<5, 2> split_value1;
    int feature_index2_left;  // Second split for left branch
    int feature_index2_right; // Second split for right branch
    flx::floatx<5, 2> split_value2_left;
    flx::floatx<5, 2> split_value2_right;
    flx::floatx<5, 2> left_left_value;   // Leaf for left-left
    flx::floatx<5, 2> left_right_value;  // Leaf for left-right
    flx::floatx<5, 2> right_left_value;  // Leaf for right-left
    flx::floatx<5, 2> right_right_value; // Leaf for right-right

    flx::floatx<5, 2> predict(const double features[]) {
        if (features[feature_index1] < split_value1) {
            // Left branch
            if (feature_index2_left == -1) return left_left_value;
            return features[feature_index2_left] < split_value2_left ? left_left_value : left_right_value;
        } else {
            // Right branch
            if (feature_index2_right == -1) return right_left_value;
            return features[feature_index2_right] < split_value2_right ? right_left_value : right_right_value;
        }
    }

    void fit(const DataPoint data[], const double weights[], int n_samples) {
        flx::floatx<5, 2> best_error = 9999.0;
        
        // First split
        for (int f1 = 0; f1 < N_FEATURES; ++f1) {
            flx::floatx<5, 2> values[MAX_SAMPLES];
            for (int i = 0; i < n_samples; ++i) values[i] = data[i].features[f1];
            std::sort(values, values + n_samples);
            
            for (int i = 0; i < n_samples - 1; ++i) {
                flx::floatx<5, 2> split1 = (values[i] + values[i + 1]) / 2;
                flx::floatx<5, 2> left_sum = 0.0, right_sum = 0.0;
                flx::floatx<5, 2> left_weight = 0.0, right_weight = 0.0;
                int left_count = 0, right_count = 0;
                
                for (int j = 0; j < n_samples; ++j) {
                    if (data[j].features[f1] < split1) {
                        left_sum += weights[j] * data[j].target;
                        left_weight += weights[j];
                        left_count++;
                    } else {
                        right_sum += weights[j] * data[j].target;
                        right_weight += weights[j];
                        right_count++;
                    }
                }
                flx::floatx<5, 2> temp = 0.0;
                flx::floatx<5, 2> left_val = left_weight > 1e-10 ? left_sum / left_weight : temp;
                flx::floatx<5, 2> right_val = right_weight > 1e-10 ? right_sum / right_weight : temp;
                
                flx::floatx<5, 2> left_left_val = left_val, left_right_val = left_val; // Second split on left branch
                int f2_left = -1;
                flx::floatx<5, 2> split2_left = 0.0;
                flx::floatx<5, 2> left_error = 0.0;
                if (left_count > 1) {
                    flx::floatx<5, 2> best_left_error = std::numeric_limits<flx::floatx<5, 2>>::infinity();
                    for (int f2 = 0; f2 < N_FEATURES; ++f2) {
                        for (int k = 0; k < n_samples - 1; ++k) {
                            flx::floatx<5, 2> split2 = (values[k] + values[k + 1]) / 2;
                            flx::floatx<5, 2> ll_sum = 0.0, lr_sum = 0.0;
                            flx::floatx<5, 2> ll_weight = 0.0, lr_weight = 0.0;
                            
                            for (int j = 0; j < n_samples; ++j) {
                                if (data[j].features[f1] < split1 && data[j].features[f2] < split2) {
                                    ll_sum += weights[j] * data[j].target;
                                    ll_weight += weights[j];
                                } else if (data[j].features[f1] < split1) {
                                    lr_sum += weights[j] * data[j].target;
                                    lr_weight += weights[j];
                                }
                            }
                            
                            flx::floatx<5, 2> temp = 0.0;
                            flx::floatx<5, 2> ll_val = ll_weight > 1e-10 ? ll_sum / ll_weight : temp;
                            flx::floatx<5, 2> lr_val = lr_weight > 1e-10 ? lr_sum / lr_weight : temp;
                            flx::floatx<5, 2> error = 0.0;
                            for (int j = 0; j < n_samples; ++j) {
                                if (data[j].features[f1] < split1) {
                                    flx::floatx<5, 2> pred = data[j].features[f2] < split2 ? ll_val : lr_val;
                                    flx::floatx<5, 2> resid = data[j].target - pred;
                                    error += weights[j] * resid * resid;
                                }
                            }
                            if (error < best_left_error) {
                                best_left_error = error;
                                f2_left = f2;
                                split2_left = split2;
                                left_left_val = ll_val;
                                left_right_val = lr_val;
                            }
                        }
                    }
                    left_error = best_left_error;
                }
                
                // Second split on right branch
                flx::floatx<5, 2> right_left_val = right_val, right_right_val = right_val;
                int f2_right = -1;
                flx::floatx<5, 2> split2_right = 0.0;
                flx::floatx<5, 2> right_error = 0.0;
                if (right_count > 1) {
                    flx::floatx<5, 2> best_right_error = std::numeric_limits<float>::infinity();
                    for (int f2 = 0; f2 < N_FEATURES; ++f2) {
                        for (int k = 0; k < n_samples - 1; ++k) {
                            flx::floatx<5, 2> split2 = (values[k] + values[k + 1]) / 2;
                            float rl_sum = 0.0, rr_sum = 0.0;
                            float rl_weight = 0.0, rr_weight = 0.0;
                            
                            for (int j = 0; j < n_samples; ++j) {
                                if (data[j].features[f1] >= split1 && data[j].features[f2] < split2) {
                                    rl_sum += weights[j] * data[j].target;
                                    rl_weight += weights[j];
                                } else if (data[j].features[f1] >= split1) {
                                    rr_sum += weights[j] * data[j].target;
                                    rr_weight += weights[j];
                                }
                            }
                            float temp = 0.0;
                            flx::floatx<5, 2> rl_val = rl_weight > 1e-10 ? rl_sum / rl_weight : temp;
                            flx::floatx<5, 2> rr_val = rr_weight > 1e-10 ? rr_sum / rr_weight : temp;
                            flx::floatx<5, 2> error = 0.0;
                            for (int j = 0; j < n_samples; ++j) {
                                if (data[j].features[f1] >= split1) {
                                    flx::floatx<5, 2> pred = data[j].features[f2] < split2 ? rl_val : rr_val;
                                    flx::floatx<5, 2> resid = data[j].target - pred;
                                    error += weights[j] * resid * resid;
                                }
                            }
                            if (error < best_right_error) {
                                best_right_error = error;
                                f2_right = f2;
                                split2_right = split2;
                                right_left_val = rl_val;
                                right_right_val = rr_val;
                            }
                        }
                    }
                    right_error = best_right_error;
                }
                
                flx::floatx<5, 2> total_error = left_error + right_error;
                if (total_error < best_error) {
                    best_error = total_error;
                    feature_index1 = f1;
                    split_value1 = split1;
                    feature_index2_left = f2_left;
                    feature_index2_right = f2_right;
                    split_value2_left = split2_left;
                    split_value2_right = split2_right;
                    left_left_value = left_left_val;
                    left_right_value = left_right_val;
                    right_left_value = right_left_val;
                    right_right_value = right_right_val;
                }
            }
        }
    }
};

class AdaBoostRegressor {
private:
    DecisionTree trees[MAX_ESTIMATORS];
    flx::floatx<5, 2> tree_weights[MAX_ESTIMATORS];
    int n_estimators;
    int n_trees;

public:
    AdaBoostRegressor(int n_est = MAX_ESTIMATORS) : n_estimators(n_est), n_trees(0) {}
    
    void fit(const DataPoint data[], int n_samples) {
        if (n_samples == 0) return;
        double weights[MAX_SAMPLES];
        flx::floatx<5, 2> predictions[MAX_SAMPLES];
        for (int i = 0; i < n_samples; ++i) {
            weights[i] = 1.0 / n_samples;
            predictions[i] = 0.0;
        }

        for (int t = 0; t < n_estimators && n_trees < MAX_ESTIMATORS; ++t) {
            DecisionTree tree;
            tree.fit(data, weights, n_samples);
            
            // Compute loss (normalized squared error)
            flx::floatx<5, 2> max_loss = 0.0;
            flx::floatx<5, 2> loss_sum = 0.0;
            double losses[MAX_SAMPLES];
            for (int i = 0; i < n_samples; ++i) {
                flx::floatx<5, 2> pred = tree.predict(data[i].features);
                flx::floatx<5, 2> resid = abs(data[i].target - pred);
                losses[i] = resid * resid;
                if (losses[i] > max_loss) max_loss = losses[i];
            }
            if (max_loss < 1e-10) break;
            for (int i = 0; i < n_samples; ++i) {
                losses[i] /= max_loss;
                loss_sum += weights[i] * losses[i];
            }
            flx::floatx<5, 2> error = loss_sum;
            if (error > 0.999) break;
            
            // Compute tree weight with learning rate
            float beta = error / (1.0 - error + 1e-10);
            flx::floatx<5, 2> alpha = LEARNING_RATE * 0.5 * log(1.0 / (beta + 1e-10));
            if (alpha <= 0) continue;
            
            // Update predictions and weights
            trees[n_trees] = tree;
            tree_weights[n_trees] = alpha;
            n_trees++;
            flx::floatx<5, 2> total_weight = 0.0;
            for (int i = 0; i < n_samples; ++i) {
                flx::floatx<5, 2> pred = tree.predict(data[i].features);
                predictions[i] += alpha * pred;
                weights[i] *= pow(losses[i], 1.0 - error);
                total_weight += weights[i];
            }
            for (int i = 0; i < n_samples; ++i) {
                weights[i] /= (total_weight + 1e-10);
            }
        }
        std::cout << "Trained " << n_trees << " trees" << std::endl;
    }
    
    flx::floatx<5, 2> predict(const double features[]) {
        flx::floatx<5, 2> sum = 0.0;
        flx::floatx<5, 2> weight_sum = 0.0;
        for (int i = 0; i < n_trees; ++i) {
            sum += tree_weights[i] * trees[i].predict(features);
            weight_sum += tree_weights[i];
        }
        return sum / (weight_sum + 1e-10);
    }
};

void compute_feature_stats(const DataPoint data[], int n_samples, double means[], double stds[]) {
    for (int j = 0; j < N_FEATURES; ++j) {
        means[j] = 0.0;
        stds[j] = 0.0;
    }
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
            flx::floatx<5, 2> diff = data[i].features[j] - means[j];
            stds[j] += diff * diff;
        
        }
    }
    for (int j = 0; j < N_FEATURES; ++j) {
        stds[j] = sqrt(stds[j] / n_samples);
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }
}

void transform_features(DataPoint data[], int n_samples) {
    // Apply log-transformation to CRIM (0), ZN (1), and LSTAT (11 after dropping RAD)
    int indices[] = {0, 1, 11};
    for (int i = 0; i < n_samples; ++i) {
        for (int j : indices) {
            data[i].features[j] = log(data[i].features[j] + 1e-10);
        }
    }
}

int remove_outliers(DataPoint data[], int n_samples, int& new_n_samples) {
    double means[N_FEATURES];
    double stds[N_FEATURES];
    compute_feature_stats(data, n_samples, means, stds);
    
    DataPoint temp_data[MAX_SAMPLES];
    new_n_samples = 0;
    for (int i = 0; i < n_samples; ++i) {
        bool is_outlier = false;
        for (int j = 0; j < N_FEATURES; ++j) {
            flx::floatx<5, 2> z = abs((data[i].features[j] - means[j]) / stds[j]);
            if (z > 3.0) {
                is_outlier = true;
                break;
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
    double means[N_FEATURES];
    double stds[N_FEATURES];
    compute_feature_stats(data, n_samples, means, stds);
    
    for (int i = 0; i < n_samples; ++i) {
        scaled_data[i] = data[i];
        for (int j = 0; j < N_FEATURES; ++j) {
            scaled_data[i].features[j] = (scaled_data[i].features[j] - means[j]) / stds[j];
        }
    }
}

void shuffle_data(DataPoint data[], int n_samples) {
    std::mt19937 gen(42);  // Match random_state=42
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
    
    n_samples = 0;
    int line_num = 1;
    while (getline(file, line) && n_samples < MAX_SAMPLES) {
        std::stringstream ss(line);
        std::string value;
        flx::floatx<5, 2> features[13];  // Temporary for 13 features
        double target = 0.0;
        int column = 0;
        
        while (getline(ss, value, ',') && column < 14) {
            try {
                if (value == "NA") {
                    features[column] = std::numeric_limits<float>::quiet_NaN();
                } else if (column < 13) {
                    features[column] = std::stod(value);
                } else {
                    target = std::stod(value);  // MEDV
                }
            } catch (const std::exception& e) {
                std::cerr << "Error parsing '" << value << "' at line " << line_num 
                          << ", column " << column << std::endl;
                return 1;
            }
            column++;
        }
        
        if (column != 14) {
            std::cerr << "Error: Expected 13 features + 1 target, got " 
                      << column << " at line " << line_num << std::endl;
            return 1;
        }
        
        int dest_idx = 0; // Skip RAD (index 8)
        for (int i = 0; i < 13; ++i) {
            if (i != 8) {  // Exclude RAD
                data[n_samples].features[dest_idx++] = features[i];
            }
        }
        data[n_samples].target = target;
        n_samples++;
        line_num++;
    }
    
    // Impute missing values with feature means
    flx::floatx<5, 2> means[N_FEATURES];
    int counts[N_FEATURES];
    for (int j = 0; j < N_FEATURES; ++j) {
        means[j] = 0.0;
        counts[j] = 0;
    }
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < N_FEATURES; ++j) {
            means[j] += data[i].features[j];
            counts[j]++;
    
        }
    }
    for (int j = 0; j < N_FEATURES; ++j) {
        flx::floatx<5, 2> temp = 0.0;
        means[j] = counts[j] > 0 ? means[j] / counts[j] : temp;
    }
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < N_FEATURES; ++j) {
            data[i].features[j] = means[j];
        }
    }
    
    std::cout << "Loaded " << n_samples << " data points with " 
              << N_FEATURES << " features each" << std::endl;
    return 0;
}

double compute_r2_score(const DataPoint data[], const double predictions[], int n_samples) {
    double mean_y = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        mean_y += data[i].target;
    }
    mean_y /= n_samples;
    
    double ss_tot = 0.0, ss_res = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        double diff = data[i].target - mean_y;
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
    
   
    DataPoint scaled_data[MAX_SAMPLES];  // Scale features
    scale_features(raw_data, scaled_data, n_samples);
    
    int train_size = static_cast<int>(0.8 * n_samples); // 70-30 split
    int test_size = n_samples - train_size;
    std::cout << "start:" << std::endl;
    AdaBoostRegressor ada(MAX_ESTIMATORS);
    ada.fit(scaled_data, train_size);
    
    double predictions[MAX_SAMPLES];
    flx::floatx<5, 2> check_predictions[test_size];
    flx::floatx<5, 2> mse = 0.0;
    for (int i = 0; i < test_size; ++i) {
        predictions[i] = ada.predict(scaled_data[train_size + i].features);
        check_predictions[i] = predictions[i];
        flx::floatx<5, 2> diff = predictions[i] - scaled_data[train_size + i].target;
        mse += diff * diff;
    }

    mse /= test_size;
    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
    
    double r2 = compute_r2_score(&scaled_data[train_size], predictions, test_size);
    std::cout << "R^2 Score: " << r2 << std::endl;
    
    
    return 0;
}