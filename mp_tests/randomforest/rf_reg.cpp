#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <memory>

const int MAX_DATA_POINTS = 1000; // Maximum number of data points
const int NUM_FEATURES = 13;      // CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT
const int MAX_FEATURE_INDICES = NUM_FEATURES;
const int MAX_TREES = 200;

struct DataPoint {
    __PROMISE__ features[NUM_FEATURES];
    __PROMISE__ target;
};

struct DecisionTreeRegressor {
    struct Node {
        bool is_leaf = false;
        __PROMISE__ value = 0.0;
        __PROMISE__ split_value = 0.0;
        int feature_index = -1;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
    };

    std::unique_ptr<Node> root;
    int max_depth;

    __PROMISE__ calculate_variance(const DataPoint* data, int size) {
        if (size == 0) return 0.0;
        __PROMISE__ mean = 0.0;
        for (int i = 0; i < size; ++i) mean += data[i].target;
        mean /= size;

        __PROMISE__ variance = 0.0;
        for (int i = 0; i < size; ++i) {
            __PROMISE__ diff = data[i].target - mean;
            variance += diff * diff;
        }
        return variance / size;
    }

    void sort_values(__PROMISE__* values, int size) {
        for (int i = 0; i < size - 1; ++i) {
            for (int j = i + 1; j < size; ++j) {
                if (values[i] > values[j]) {
                    __PROMISE__ temp = values[i];
                    values[i] = values[j];
                    values[j] = temp;
                }
            }
        }
    }

    std::pair<int, __PROMISE__> find_best_split(const DataPoint* data, int size, const int* feature_indices, int num_indices) {
        if (size == 0 || num_indices == 0) return {-1, 0.0};

        __PROMISE__ best_reduction = -9999.0;
        int best_feature = -1;
        __PROMISE__ best_value = 0.0;

        __PROMISE__ total_variance = calculate_variance(data, size);

        for (int idx = 0; idx < num_indices; ++idx) {
            int f = feature_indices[idx];
            __PROMISE__ values[MAX_DATA_POINTS];
            for (int i = 0; i < size; ++i) {
                if (f >= NUM_FEATURES) return {-1, 0.0};
                values[i] = data[i].features[f];
            }
            sort_values(values, size);
            if (size < 2) continue;

            for (int i = 0; i < size - 1; ++i) {
                __PROMISE__ split_val = (values[i] + values[i + 1]) / 2;
                DataPoint left_data[MAX_DATA_POINTS];
                DataPoint right_data[MAX_DATA_POINTS];
                int left_size = 0, right_size = 0;

                for (int j = 0; j < size; ++j) {
                    if (data[j].features[f] < split_val) {
                        left_data[left_size++] = data[j];
                    } else {
                        right_data[right_size++] = data[j];
                    }
                }
                if (left_size == 0 || right_size == 0) continue;

                __PROMISE__ left_var = calculate_variance(left_data, left_size);
                __PROMISE__ right_var = calculate_variance(right_data, right_size);
                __PROMISE__ reduction = total_variance - 
                                  (left_size * left_var + right_size * right_var) / size;

                if (reduction > best_reduction) {
                    best_reduction = reduction;
                    best_feature = f;
                    best_value = split_val;
                }
            }
        }
        return {best_feature, best_value};
    }

    std::unique_ptr<Node> build_tree(const DataPoint* data, int size, const int* feature_indices, int num_indices, int depth) {
        auto node = std::make_unique<Node>();

        if (size == 0) {
            node->is_leaf = true;
            node->value = 0.0;
            return node;
        }

        if (depth >= max_depth || size < 2) {
            node-> is_leaf = true;
            __PROMISE__ sum = 0.0;
            for (int i = 0; i < size; ++i) sum += data[i].target;
            node->value = sum / size;
            return node;
        }

        auto [feature, value] = find_best_split(data, size, feature_indices, num_indices);
        if (feature == -1) {
            node->is_leaf = true;
            __PROMISE__ sum = 0.0;
            for (int i = 0; i < size; ++i) sum += data[i].target;
            node->value = sum / size;
            return node;
        }

        DataPoint left_data[MAX_DATA_POINTS];
        DataPoint right_data[MAX_DATA_POINTS];
        int left_size = 0, right_size = 0;

        for (int i = 0; i < size; ++i) {
            if (data[i].features[feature] < value) {
                left_data[left_size++] = data[i];
            } else {
                right_data[right_size++] = data[i];
            }
        }

        node->feature_index = feature;
        node->split_value = value;
        node->left = build_tree(left_data, left_size, feature_indices, num_indices, depth + 1);
        node->right = build_tree(right_data, right_size, feature_indices, num_indices, depth + 1);

        return node;
    }

public:
    DecisionTreeRegressor(int max_d = 15) : max_depth(max_d) {}

    void fit(const DataPoint* data, int size, const int* feature_indices, int num_indices) {
        if (size == 0) {
            std::cerr << "Error: Empty dataset in DecisionTreeRegressor::fit" << std::endl;
            return;
        }
        root = build_tree(data, size, feature_indices, num_indices, 0);
    }

    __PROMISE__ predict(const __PROMISE__* features) {
        if (!root) return 0.0;
        Node* current = root.get();
        while (!current->is_leaf) {
            if (current->feature_index >= NUM_FEATURES) return 0.0;
            current = (features[current->feature_index] < current->split_value) ? 
                      current->left.get() : current->right.get();
            if (!current) return 0.0;
        }
        return current->value;
    }
};

class RandomForestRegressor {
private:
    DecisionTreeRegressor trees[MAX_TREES];
    int n_trees;
    int max_depth;
    int max_features;
    unsigned int seed;

    void bootstrap_sample(const DataPoint* data, int size, DataPoint* sample, int& sample_size, std::mt19937& gen) {
        if (size == 0) {
            sample_size = 0;
            return;
        }
        std::uniform_int_distribution<> dis(0, size - 1);
        sample_size = size;
        for (int i = 0; i < size; ++i) {
            sample[i] = data[dis(gen)];
        }
    }

    void random_features(int* feature_indices, int& num_indices, std::mt19937& gen) {
        int all_features[NUM_FEATURES];
        for (int i = 0; i < NUM_FEATURES; ++i) all_features[i] = i;
        for (int i = NUM_FEATURES - 1; i > 0; --i) {
            std::uniform_int_distribution<> dis(0, i);
            int j = dis(gen);
            std::swap(all_features[i], all_features[j]);
        }
        num_indices = std::min(max_features, NUM_FEATURES);
        for (int i = 0; i < num_indices; ++i) {
            feature_indices[i] = all_features[i];
        }
    }

public:
    RandomForestRegressor(int n_t = 200, int m_d = 15, int m_f = -1, unsigned int s = 42)
        : n_trees(n_t), max_depth(m_d), max_features(m_f), seed(s) {}

    void fit(const DataPoint* data, int size) {
        if (size == 0) {
            std::cerr << "Error: Empty dataset in RandomForestRegressor::fit" << std::endl;
            return;
        }
        std::mt19937 gen(seed);
        if (max_features <= 0) max_features = static_cast<int>(sqrt(NUM_FEATURES)) + 1;

        for (int i = 0; i < n_trees; ++i) {
            DataPoint sample[MAX_DATA_POINTS];
            int sample_size = 0;
            bootstrap_sample(data, size, sample, sample_size, gen);
            int feature_indices[MAX_FEATURE_INDICES];
            int num_indices = 0;
            random_features(feature_indices, num_indices, gen);
            if (sample_size == 0 || num_indices == 0) continue;
            trees[i] = DecisionTreeRegressor(max_depth);
            trees[i].fit(sample, sample_size, feature_indices, num_indices);
        }
    }

    __PROMISE__ predict(const __PROMISE__* features) {
        if (n_trees == 0) return 0.0;
        __PROMISE__ sum = 0.0;
        int valid_trees = 0;
        for (int i = 0; i < n_trees; ++i) {
            __PROMISE__ pred = trees[i].predict(features);
            sum += pred;
            valid_trees++;
        }
        double temp = 0.0;
        return valid_trees > 0 ? sum / valid_trees : temp;
    }
};

// Compute feature statistics (mean and std) for preprocessing
void compute_feature_stats(const DataPoint* data, int size, __PROMISE__ means[], __PROMISE__ stds[]) {
    for (int j = 0; j < NUM_FEATURES; ++j) {
        means[j] = 0.0;
        stds[j] = 0.0;
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            means[j] += data[i].features[j];
        }
    }
    for (int j = 0; j < NUM_FEATURES; ++j) {
        means[j] /= size;
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            __PROMISE__ diff = data[i].features[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    for (int j = 0; j < NUM_FEATURES; ++j) {
        stds[j] = sqrt(stds[j] / size);
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }
}

// Log-transformation for CRIM (0), ZN (1), and LSTAT (12)
void transform_features(DataPoint* data, int size) {
    int indices[] = {0, 1, 12}; // CRIM, ZN, LSTAT
    for (int i = 0; i < size; ++i) {
        for (int j : indices) {
            if (data[i].features[j] > 0) {
                data[i].features[j] = log(data[i].features[j] + 1e-10);
            }
        }
    }
}

// Remove outliers based on z-score > 3
int remove_outliers(DataPoint* data, int size, int& new_size) {
    __PROMISE__ means[NUM_FEATURES];
    __PROMISE__ stds[NUM_FEATURES];
    compute_feature_stats(data, size, means, stds);

    DataPoint temp_data[MAX_DATA_POINTS];
    new_size = 0;
    for (int i = 0; i < size; ++i) {
        bool is_outlier = false;
        for (int j = 0; j < NUM_FEATURES; ++j) {
            __PROMISE__ z = abs((data[i].features[j] - means[j]) / stds[j]);
            if (z > 3.0) {
                is_outlier = true;
                break;
            }
        }
        if (!is_outlier) {
            temp_data[new_size++] = data[i];
        }
    }

    for (int i = 0; i < new_size; ++i) {
        data[i] = temp_data[i];
    }
    return new_size > 0 ? 0 : 1;
}

// Scale features (standardization)
void scale_features(const DataPoint* data, int size, DataPoint* scaled_data) {
    if (size == 0) return;
    __PROMISE__ means[NUM_FEATURES];
    __PROMISE__ stds[NUM_FEATURES];
    compute_feature_stats(data, size, means, stds);

    for (int i = 0; i < size; ++i) {
        scaled_data[i] = data[i];
        for (int j = 0; j < NUM_FEATURES; ++j) {
            scaled_data[i].features[j] = (data[i].features[j] - means[j]) / stds[j];
        }
    }
}

// Shuffle data
void shuffle_data(DataPoint* data, int size) {
    std::mt19937 gen(42); // Consistent with AdaBoost
    for (int i = size - 1; i > 0; --i) {
        std::uniform_int_distribution<> dis(0, i);
        int j = dis(gen);
        DataPoint temp = data[i];
        data[i] = data[j];
        data[j] = temp;
    }
}

// Read CSV with missing value handling
int read_csv(const std::string& filename, DataPoint* data) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return 0;
    }

    std::string line;
    getline(file, line); // Skip header

    int size = 0;
    __PROMISE__ feature_sums[NUM_FEATURES] = {0.0};
    int feature_counts[NUM_FEATURES] = {0};

    // First pass: read data and compute sums for missing value imputation
    while (getline(file, line) && size < MAX_DATA_POINTS) {
        std::stringstream ss(line);
        std::string value;
        DataPoint point;

        for (int i = 0; i < NUM_FEATURES; ++i) {
            getline(ss, value, ',');
            if (value == "NA") {
                point.features[i] = 0.0; // Temporary placeholder
            } else {
                point.features[i] = std::stod(value);
                feature_sums[i] += point.features[i];
                feature_counts[i]++;
            }
        }
        getline(ss, value); // Read target (MEDV)
        point.target = std::stod(value);
        data[size++] = point;
    }

    // Compute means for missing values
    __PROMISE__ feature_means[NUM_FEATURES];
    for (int i = 0; i < NUM_FEATURES; ++i) {
        double temp = 0.0;
        feature_means[i] = feature_counts[i] > 0 ? feature_sums[i] / feature_counts[i] : temp;
    }

    // Second pass: replace NA with mean
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            if (data[i].features[j] == 0.0 && feature_counts[j] < size) {
                data[i].features[j] = feature_means[j];
            }
        }
    }

    std::cout << "Loaded " << size << " data points with " << NUM_FEATURES << " features each" << std::endl;
    file.close();
    return size;
}

// Write predictions to CSV
void write_predictions(const DataPoint* data, int size, const __PROMISE__* predictions, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing" << std::endl;
        return;
    }
    file << "CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT,MEDV,prediction\n";

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            file << data[i].features[j];
            if (j < NUM_FEATURES - 1) file << ",";
        }
        file << "," << data[i].target << "," << predictions[i] << "\n";
    }
    file.close();
}

// Compute R² score
__PROMISE__ compute_r2_score(const DataPoint* data, const __PROMISE__* predictions, int size) {
    __PROMISE__ mean_y = 0.0;
    for (int i = 0; i < size; ++i) {
        mean_y += data[i].target;
    }
    mean_y /= size;

    __PROMISE__ ss_tot = 0.0, ss_res = 0.0;
    for (int i = 0; i < size; ++i) {
        __PROMISE__ diff = data[i].target - mean_y;
        ss_tot += diff * diff;
        diff = data[i].target - predictions[i];
        ss_res += diff * diff;
    }
    return 1.0 - (ss_res / (ss_tot + 1e-10));
}

int main() {
    DataPoint raw_data[MAX_DATA_POINTS];
    int data_size = read_csv("boston_housing.csv", raw_data);
    if (data_size == 0) {
        std::cerr << "Error: No valid data loaded from CSV" << std::endl;
        return 1;
    }

    // Apply log-transformation
    transform_features(raw_data, data_size);
    std::cout << "Applied log-transformation to CRIM, ZN, LSTAT" << std::endl;

    // Remove outliers
    int new_size = 0;
    if (remove_outliers(raw_data, data_size, new_size)) {
        std::cerr << "Error: No data left after outlier removal" << std::endl;
        return 1;
    }
    data_size = new_size;
    std::cout << "After outlier removal: " << data_size << " data points" << std::endl;

    // Shuffle data
    shuffle_data(raw_data, data_size);
    std::cout << "Data shuffled" << std::endl;

    // Scale features
    DataPoint scaled_data[MAX_DATA_POINTS];
    scale_features(raw_data, data_size, scaled_data);
    std::cout << "Features scaled" << std::endl;

    // Train-test split
    int train_size = static_cast<int>(0.8 * data_size);
    if (train_size == 0) {
        std::cerr << "Error: Dataset too small for train-test split" << std::endl;
        return 1;
    }
    DataPoint* train_data = scaled_data;
    DataPoint* test_data = scaled_data + train_size;
    int test_size = data_size - train_size;

    // Train random forest
    unsigned int random_seed = 42;
    RandomForestRegressor rf(10, 2, 10, random_seed);
    auto start = std::chrono::high_resolution_clock::now();
    rf.fit(train_data, train_size);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Training time: " << duration.count() << " ms" << std::endl;

    // Evaluate
    __PROMISE__ predictions[MAX_DATA_POINTS];
    __PROMISE__ check_predictions[test_size];
    __PROMISE__ mse = 0.0;
    for (int i = 0; i < test_size; ++i) {
        predictions[i] = rf.predict(test_data[i].features);
        check_predictions[i] = predictions[i];
        __PROMISE__ diff = predictions[i] - test_data[i].target;
        mse += diff * diff;
    }
    mse /= test_size;
    std::cout << "Mean Squared Error (MSE): " << mse << std::endl;
    PROMISE_CHECK_ARRAY(check_predictions, test_size);
    double r2 = compute_r2_score(test_data, predictions, test_size);
    std::cout << "R^2 Score: " << r2 << std::endl;

    // Write predictions
    write_predictions(test_data, test_size, predictions, "../results/randomforest/pred_boston.csv");

    return 0;
}