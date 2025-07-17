#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <vector>
#include <string>

const int MAX_FEATURES = 100;
const int MAX_DATA_POINTS = 10000;
const int MAX_FEATURE_INDICES = 100;


const std::vector<std::string> FEATURE_NAMES = {
    // Boston Housing Dataset feature names
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
};

struct DataPoint {
    double* features;
    int num_features;
    double target;

    DataPoint() : features(nullptr), num_features(0), target(0.0) {}

    ~DataPoint() { delete[] features; }

    DataPoint(const DataPoint& other) : num_features(other.num_features), target(other.target) {
        features = new double[num_features];
        for (int i = 0; i < num_features; ++i) {
            features[i] = other.features[i];
        }
    }

    DataPoint& operator=(const DataPoint& other) {
        if (this != &other) {
            delete[] features;
            num_features = other.num_features;
            target = other.target;
            features = new double[num_features];
            for (int i = 0; i < num_features; ++i) {
                features[i] = other.features[i];
            }
        }
        return *this;
    }
};

struct DecisionTreeRegressor {
    struct Node {
        bool is_leaf = false;
        double value = 0.0;
        double split_value = 0.0;
        int feature_index = -1;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    std::unique_ptr<Node> root;
    int max_depth;

    double calculate_variance(const DataPoint* data, int size) {
        if (size == 0) return 0.0;
        double mean = 0.0;
        for (int i = 0; i < size; ++i) mean += data[i].target;
        mean /= size;

        double variance = 0.0;
        for (int i = 0; i < size; ++i) {
            double diff = data[i].target - mean;
            variance += diff * diff;
        }
        return variance / size;
    }

    std::pair<int, double> find_best_split(const DataPoint* data, int data_size,
                                          const int* feature_indices, int num_features) {
        if (data_size == 0 || num_features == 0) return {-1, 0.0};

        double best_reduction = -std::numeric_limits<double>::infinity();
        int best_feature = -1;
        double best_value = 0.0;

        double total_variance = calculate_variance(data, data_size);

        for (int f_idx = 0; f_idx < num_features; ++f_idx) {
            int f = feature_indices[f_idx];
            double* values = new double[data_size];
            int valid_values = 0;
            for (int i = 0; i < data_size; ++i) {
                if (f >= data[i].num_features) {
                    std::cerr << "Error: Feature index " << f << " out of bounds" << std::endl;
                    delete[] values;
                    return {-1, 0.0};
                }
                values[valid_values++] = data[i].features[f];
            }
            std::sort(values, values + valid_values);
            if (valid_values < 2) {
                delete[] values;
                continue;
            }

            for (int i = 0; i < valid_values - 1; ++i) {
                double split_val = (values[i] + values[i + 1]) / 2;

                DataPoint* left = new DataPoint[data_size];
                DataPoint* right = new DataPoint[data_size];
                int left_size = 0, right_size = 0;

                for (int j = 0; j < data_size; ++j) {
                    if (data[j].features[f] < split_val) {
                        left[left_size++] = data[j];
                    } else {
                        right[right_size++] = data[j];
                    }
                }

                if (left_size == 0 || right_size == 0) {
                    delete[] left;
                    delete[] right;
                    continue;
                }

                double left_var = calculate_variance(left, left_size);
                double right_var = calculate_variance(right, right_size);
                double reduction = total_variance -
                                  (left_size * left_var + right_size * right_var) / data_size;

                if (reduction > best_reduction) {
                    best_reduction = reduction;
                    best_feature = f;
                    best_value = split_val;
                }

                delete[] left;
                delete[] right;
            }
            delete[] values;
        }
        return {best_feature, best_value};
    }

    std::unique_ptr<Node> build_tree(const DataPoint* data, int data_size,
                                    const int* feature_indices, int num_features, int depth) {
        auto node = std::make_unique<Node>();

        if (data_size == 0) {
            std::cerr << "Error: Empty data in build_tree at depth " << depth << std::endl;
            node->is_leaf = true;
            node->value = 0.0;
            return node;
        }

        if (depth >= max_depth || data_size < 2) {
            node->is_leaf = true;
            double sum = 0.0;
            for (int i = 0; i < data_size; ++i) sum += data[i].target;
            node->value = sum / data_size;
            return node;
        }

        auto [feature, value] = find_best_split(data, data_size, feature_indices, num_features);
        if (feature == -1) {
            node->is_leaf = true;
            double sum = 0.0;
            for (int i = 0; i < data_size; ++i) sum += data[i].target;
            node->value = sum / data_size;
            return node;
        }

        DataPoint* left_data = new DataPoint[data_size];
        DataPoint* right_data = new DataPoint[data_size];
        int left_size = 0, right_size = 0;

        for (int i = 0; i < data_size; ++i) {
            if (data[i].features[feature] < value) {
                left_data[left_size++] = data[i];
            } else {
                right_data[right_size++] = data[i];
            }
        }

        node->feature_index = feature;
        node->split_value = value;
        node->left = build_tree(left_data, left_size, feature_indices, num_features, depth + 1);
        node->right = build_tree(right_data, right_size, feature_indices, num_features, depth + 1);

        delete[] left_data;
        delete[] right_data;

        return node;
    }

public:
    DecisionTreeRegressor(int max_d = 10) : max_depth(max_d) {}

    void fit(const DataPoint* data, int data_size, const int* feature_indices, int num_features) {
        if (data_size == 0) {
            std::cerr << "Error: Empty dataset in DecisionTreeRegressor::fit" << std::endl;
            return;
        }
        root = build_tree(data, data_size, feature_indices, num_features, 0);
    }

    double predict(const double* features, int num_features) {
        if (!root) {
            std::cerr << "Error: Tree not initialized in predict" << std::endl;
            return 0.0;
        }
        Node* current = root.get();
        while (!current->is_leaf) {
            if (current->feature_index >= num_features) {
                std::cerr << "Error: Feature index " << current->feature_index
                          << " exceeds feature size " << num_features << std::endl;
                return 0.0;
            }
            current = (features[current->feature_index] < current->split_value) ?
                      current->left.get() : current->right.get();
            if (!current) {
                std::cerr << "Error: Null node in predict" << std::endl;
                return 0.0;
            }
        }
        return current->value;
    }
};


DataPoint* scale_features(const DataPoint* data, int data_size, int& out_size) {
    if (data_size == 0) {
        std::cerr << "Error: Empty data in scale_features" << std::endl;
        out_size = 0;
        return nullptr;
    }

    int n_features = data[0].num_features;
    DataPoint* scaled_data = new DataPoint[data_size];
    for (int i = 0; i < data_size; ++i) {
        scaled_data[i] = data[i];
    }

    double* means = new double[n_features]();
    double* stds = new double[n_features]();
    int* counts = new int[n_features](); // Track non-missing values

    // Compute means
    for (int i = 0; i < data_size; ++i) {
        if (data[i].num_features != n_features) {
            std::cerr << "Error: Inconsistent feature count" << std::endl;
            delete[] scaled_data;
            delete[] means;
            delete[] stds;
            delete[] counts;
            out_size = 0;
            return nullptr;
        }
        for (int j = 0; j < n_features; ++j) {
            if (!std::isnan(data[i].features[j])) {
                means[j] += data[i].features[j];
                counts[j]++;
            }
        }
    }
    for (int j = 0; j < n_features; ++j) {
        means[j] = (counts[j] > 0) ? means[j] / counts[j] : 0.0;
    }

    // Compute standard deviations
    for (int i = 0; i < data_size; ++i) {
        for (int j = 0; j < n_features; ++j) {
            if (!std::isnan(data[i].features[j])) {
                double diff = data[i].features[j] - means[j];
                stds[j] += diff * diff;
            }
        }
    }
    for (int j = 0; j < n_features; ++j) {
        stds[j] = (counts[j] > 0) ? std::sqrt(stds[j] / counts[j]) : 1e-9;
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }

    // Scale features and impute missing values with mean
    for (int i = 0; i < data_size; ++i) {
        for (int j = 0; j < n_features; ++j) {
            if (std::isnan(scaled_data[i].features[j])) {
                scaled_data[i].features[j] = 0.0; // Mean after scaling
            } else {
                scaled_data[i].features[j] = (scaled_data[i].features[j] - means[j]) / stds[j];
            }
        }
    }

    delete[] means;
    delete[] stds;
    delete[] counts;
    out_size = data_size;
    return scaled_data;
}

DataPoint* read_csv(const std::string& filename, int& out_size) {
    DataPoint* data = new DataPoint[MAX_DATA_POINTS];
    int data_size = 0;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        delete[] data;
        out_size = 0;
        return nullptr;
    }

    std::string line;
    // Read header
    if (!getline(file, line)) {
        std::cerr << "Error: Empty file or failed to read header" << std::endl;
        delete[] data;
        out_size = 0;
        return nullptr;
    }

   
    std::stringstream header_ss(line); // Validate header (optional, assuming correct format)
    std::string header_value;
    std::vector<std::string> headers;
    while (getline(header_ss, header_value, ',')) {
        headers.push_back(header_value);
    }
    if (headers.size() != 14) {
        std::cerr << "Error: Expected 14 columns, got " << headers.size() << std::endl;
        delete[] data;
        out_size = 0;
        return nullptr;
    }

    while (getline(file, line) && data_size < MAX_DATA_POINTS) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        double target = 0.0;

        std::vector<std::string> values;
        while (getline(ss, value, ',')) {
            values.push_back(value);
        }

        // Check for correct number of columns
        if (values.size() != 14) {
            std::cerr << "Warning: Skipping row with " << values.size()
                      << " columns (expected 14)" << std::endl;
            continue;
        }

        // Parse features (first 13 columns)
        bool valid_row = true;
        for (int i = 0; i < 13; ++i) {
            if (values[i] == "NA") {
                features.push_back(std::nan("")); // Use NaN for missing values
            } else {
                try {
                    features.push_back(std::stod(values[i]));
                } catch (...) {
                    std::cerr << "Warning: Invalid feature value in row " << data_size + 1
                              << ", column " << i + 1 << std::endl;
                    valid_row = false;
                    break;
                }
            }
        }

        // Parse target (MEDV, last column)
        if (valid_row) {
            if (values[13] == "NA") {
                std::cerr << "Warning: Missing target value in row " << data_size + 1 << std::endl;
                valid_row = false;
            } else {
                try {
                    target = std::stod(values[13]);
                } catch (...) {
                    std::cerr << "Warning: Invalid target value in row " << data_size + 1 << std::endl;
                    valid_row = false;
                }
            }
        }

        if (!valid_row) {
            continue;
        }
        
        DataPoint point;
        point.num_features = features.size();
        point.features = new double[point.num_features];
        for (int i = 0; i < point.num_features; ++i) {
            point.features[i] = features[i];
        }
        point.target = target;

        data[data_size++] = point;
    }

    file.close();
    if (data_size == 0) {
        std::cerr << "Error: No valid data points loaded" << std::endl;
        delete[] data;
        out_size = 0;
        return nullptr;
    }

    std::cout << "Loaded " << data_size << " data points with "
              << data[0].num_features << " features each" << std::endl;

    out_size = data_size;
    return data;
}

void write_predictions(const DataPoint* data, int data_size,
                      const double* predictions, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing" << std::endl;
        return;
    }

    // Write header with Boston Housing feature names
    for (size_t i = 0; i < FEATURE_NAMES.size(); ++i) {
        file << FEATURE_NAMES[i];
        if (i < FEATURE_NAMES.size() - 1) file << ",";
    }
    file << ",MEDV,prediction\n";

    for (int i = 0; i < data_size; ++i) {
        for (int j = 0; j < data[i].num_features; ++j) {
            if (std::isnan(data[i].features[j])) {
                file << "NA";
            } else {
                file << data[i].features[j];
            }
            if (j < data[i].num_features - 1) file << ",";
        }
        file << "," << data[i].target << "," << predictions[i] << "\n";
    }

    file.close();
}

int main() {
    int raw_data_size;
    DataPoint* raw_data = read_csv("../data/regression/boston_housing.csv", raw_data_size);
    if (raw_data_size == 0) {
        std::cerr << "Error: No valid data loaded from CSV" << std::endl;
        delete[] raw_data;
        return 1;
    }

    int data_size;
    DataPoint* data = scale_features(raw_data, raw_data_size, data_size);
    delete[] raw_data;
    if (data_size == 0) {
        std::cerr << "Error: Feature scaling failed" << std::endl;
        delete[] data;
        return 1;
    }

    int train_size = static_cast<int>(0.7 * data_size);
    if (train_size == 0) {
        std::cerr << "Error: Dataset too small for train-test split" << std::endl;
        delete[] data;
        return 1;
    }

    DataPoint* train_data = new DataPoint[train_size];
    DataPoint* test_data = new DataPoint[data_size - train_size];
    int test_size = data_size - train_size;

    for (int i = 0; i < train_size; ++i) {
        train_data[i] = data[i];
    }
    for (int i = 0; i < test_size; ++i) {
        test_data[i] = data[train_size + i];
    }
    delete[] data;

    int num_features = test_data[0].num_features;
    int* all_features = new int[num_features];
    for (int i = 0; i < num_features; ++i) {
        all_features[i] = i;
    }

    DecisionTreeRegressor dt(5);
    auto start = std::chrono::high_resolution_clock::now();
    dt.fit(train_data, train_size, all_features, num_features);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Training time: " << duration.count() << " ms" << std::endl;

    double* predictions = new double[test_size];
    double mse = 0.0;
    for (int i = 0; i < test_size; ++i) {
        predictions[i] = dt.predict(test_data[i].features, test_data[i].num_features);
        double diff = predictions[i] - test_data[i].target;
        mse += diff * diff;
        std::cout << "Prediction: " << predictions[i] << ", Actual: " << test_data[i].target << std::endl;
    }
    mse /= test_size;
    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;

    write_predictions(test_data, test_size, predictions, "results/decisiontree/pred_reg.csv");

    delete[] train_data;
    delete[] test_data;
    delete[] all_features;
    delete[] predictions;

    return 0;
}