#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

struct DataPoint {
    double* features;
    int true_label;
    int cluster;
};

DataPoint* scale_features(DataPoint* data, size_t data_size, size_t n_features) {
    if (data_size == 0) return data;

    double* means = new double[n_features]();
    double* stds = new double[n_features]();

    for (size_t i = 0; i < data_size; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            means[j] += data[i].features[j];
        }
    }
    for (size_t j = 0; j < n_features; ++j) {
        means[j] /= data_size;
    }

    for (size_t i = 0; i < data_size; ++i) {
        for (size_t j = 0; j < n_features; ++j) {
            float diff = data[i].features[j] - means[j];
            stds[j] += diff * diff;
        }
    }
    for (size_t j = 0; j < n_features; ++j) {
        stds[j] = sqrt(stds[j] / data_size);
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }

    DataPoint* scaled_data = new DataPoint[data_size];
    for (size_t i = 0; i < data_size; ++i) {
        scaled_data[i].features = new double[n_features];
        for (size_t j = 0; j < n_features; ++j) {
            scaled_data[i].features[j] = (data[i].features[j] - means[j]) / stds[j];
        }
        scaled_data[i].true_label = data[i].true_label;
        scaled_data[i].cluster = data[i].cluster;
    }

    delete[] means;
    delete[] stds;
    return scaled_data;
}

class DBSCAN {

public:
    DBSCAN(float epsilon = 0.01, int min_points = 5) : eps(epsilon), min_pts(min_points), data(nullptr), data_size(0), n_features(0) {}

    ~DBSCAN() {
        for (size_t i = 0; i < data_size; ++i) {
            delete[] data[i].features;
        }
        delete[] data;
    }

    void fit(DataPoint* input_data, size_t size, size_t features) {
        data = input_data;
        data_size = size;
        n_features = features;
        for (size_t i = 0; i < data_size; ++i) {
            data[i].cluster = -1;
        }

        int cluster_label = 0;
        std::unordered_set<int> visited;

        for (size_t i = 0; i < data_size; ++i) {
            if (visited.find(i) != visited.end()) continue;

            visited.insert(i);
            int neighbor_count;
            int* neighbors = find_neighbors(i, neighbor_count);

            if (neighbor_count >= min_pts) {
                expand_cluster(i, cluster_label, visited);
                cluster_label++;
            }
            delete[] neighbors;
        }
    }

    int* get_labels(size_t& label_count) const {
        label_count = data_size;
        int* labels = new int[label_count];
        for (size_t i = 0; i < data_size; ++i) {
            labels[i] = data[i].cluster;
        }
        return labels;
    }

private:
    half_float::half eps;
    int min_pts;
    DataPoint* data;
    size_t data_size;
    size_t n_features;
    int* find_neighbors(int point_idx, int& neighbor_count) {
        neighbor_count = 0;
        for (size_t i = 0; i < data_size; ++i) {
            if (i != point_idx && distance(data[point_idx].features, data[i].features) <= eps) {
                neighbor_count++;
            }
        }
        int* neighbors = new int[neighbor_count];
        int idx = 0;
        for (size_t i = 0; i < data_size; ++i) {
            if (i != point_idx && distance(data[point_idx].features, data[i].features) <= eps) {
                neighbors[idx++] = i;
            }
        }
        return neighbors;
    }
    float distance(const double* p1, const double* p2) {
        float sum = 0.0;
        for (size_t i = 0; i < n_features; ++i) {
            float diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }



    void expand_cluster(int point_idx, int cluster_label, std::unordered_set<int>& visited) {
        data[point_idx].cluster = cluster_label;
        int neighbor_count;
        int* neighbors = find_neighbors(point_idx, neighbor_count);

        if (neighbor_count < min_pts) {
            delete[] neighbors;
            return;
        }

        for (int i = 0; i < neighbor_count; ++i) {
            int neighbor_idx = neighbors[i];
            if (visited.find(neighbor_idx) == visited.end()) {
                visited.insert(neighbor_idx);
                if (data[neighbor_idx].cluster == -1) {
                    expand_cluster(neighbor_idx, cluster_label, visited);
                }
            }
        }
        delete[] neighbors;
    }

};

double adjusted_mutual_information(const int* true_labels, const int* pred_labels, size_t n) {
    if (n == 0) return 0.0;

    std::unordered_map<int, std::unordered_map<int, int>> contingency;
    for (size_t i = 0; i < n; ++i) {
        contingency[true_labels[i]][pred_labels[i]]++;
    }

    std::unordered_map<int, int> a_sums, b_sums;
    for (const auto& row : contingency) {
        for (const auto& cell : row.second) {
            a_sums[row.first] += cell.second;
            b_sums[cell.first] += cell.second;
        }
    }

    double mi = 0.0;
    float log_n = log(n);
    for (const auto& row : contingency) {
        for (const auto& cell : row.second) {
            float nij = cell.second;
            if (nij == 0) continue;
            float ai = a_sums[row.first];
            float bj = b_sums[cell.first];
            double denom = ai * bj;
            if (denom > 0) {
                mi += nij * log(nij * n / denom + 1e-10);
            }
        }
    }
    double temp_alpha = 0.0;
    mi = (mi > 0 ? mi / (n * log(2.0)) : temp_alpha);

    float h_true = 0.0;
    for (const auto& a : a_sums) {
        float p = a.second / (half_float::half)n;
        if (p > 0) h_true -= p * log(p);
    }
    h_true /= log(2.0);

    double h_pred = 0.0;
    for (const auto& b : b_sums) {
        double p = b.second / (double)n;
        if (p > 0) h_pred -= p * log(p);
    }
    h_pred /= log(2.0);

    float emi = 0.0;
    double max_mi = (h_true + h_pred) / 2.0;
    double denom = max_mi - emi + 1e-10;
    if (denom <= 0) return 0.0;
    return (mi - emi) / denom;
}

float adjusted_rand_index(const int* true_labels, const int* pred_labels, size_t n) {
    std::unordered_map<int, std::unordered_map<int, int>> contingency;
    for (size_t i = 0; i < n; ++i) {
        contingency[true_labels[i]][pred_labels[i]]++;
    }

    float sum_nij2 = 0.0;
    std::unordered_map<int, flx::floatx<4, 3>> a_sums, b_sums;
    for (const auto& row : contingency) {
        for (const auto& cell : row.second) {
            sum_nij2 += (cell.second * (cell.second - 1)) / 2.0;
            a_sums[row.first] += cell.second;
            b_sums[cell.first] += cell.second;
        }
    }

    float sum_ai2 = 0.0, sum_bj2 = 0.0;
    for (const auto& a : a_sums) sum_ai2 += (a.second * (a.second - 1)) / 2.0;
    for (const auto& b : b_sums) sum_bj2 += (b.second * (b.second - 1)) / 2.0;

    float n_choose_2 = (n * (n - 1)) / 2.0;
    float expected = (sum_ai2 * sum_bj2) / n_choose_2;
    float max_index = (sum_ai2 + sum_bj2) / 2.0;
    float index = sum_nij2;

    return (index - expected) / (max_index - expected + 1e-10);
}

DataPoint* read_csv(const std::string& filename, size_t& data_size, size_t& n_features) {
    data_size = 0;
    n_features = 0;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    std::string line;
    getline(file, line);

    while (getline(file, line)) {
        data_size++;
    }
    file.clear();
    file.seekg(0);

    getline(file, line);
    if (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        getline(ss, value, ',');
        while (getline(ss, value, ',')) {
            n_features++;
        }
        n_features--;
    }
    file.clear();
    file.seekg(0);
    getline(file, line);

    DataPoint* data = new DataPoint[data_size];
    size_t idx = 0;
    while (getline(file, line) && idx < data_size) {
        std::stringstream ss(line);
        std::string value;
        getline(ss, value, ',');

        data[idx].features = new double[n_features];
        for (size_t i = 0; i < n_features; ++i) {
            getline(ss, value, ',');
            data[idx].features[i] = std::stod(value);
        }
        getline(ss, value, ',');
        data[idx].true_label = std::stoi(value);
        data[idx].cluster = -1;
        idx++;
    }
    file.close();
    return data;
}

int main() {
    size_t data_size, n_features;
    DataPoint* raw_data = read_csv("shape_clusters_include_y.csv", data_size, n_features);
    if (!raw_data || data_size == 0) {
        std::cerr << "No data loaded. Exiting." << std::endl;
        return 1;
    }

    DataPoint* data = scale_features(raw_data, data_size, n_features);

    for (size_t i = 0; i < data_size; ++i) {
        delete[] raw_data[i].features;
    }
    delete[] raw_data;

    DBSCAN dbscan(0.1, 10);
    dbscan.fit(data, data_size, n_features);

    size_t label_count;
    int* pred_labels = dbscan.get_labels(label_count);
    int* true_labels = new int[data_size];
    for (size_t i = 0; i < data_size; ++i) {
        true_labels[i] = data[i].true_label;
    }

    int n_clusters = 0;
    for (size_t i = 0; i < label_count; ++i) {
        if (pred_labels[i] >= 0) {
            n_clusters = std::max(n_clusters, pred_labels[i] + 1);
        }
    }
    int noise_count = 0;
    for (size_t i = 0; i < label_count; ++i) {
        if (pred_labels[i] == -1) noise_count++;
    }

    double ami = adjusted_mutual_information(true_labels, pred_labels, data_size);
    
    std::cout << "Adjusted Mutual Information (AMI): " << ami << std::endl;
    // std::cout << "Adjusted Rand Index (ARI): " << ari << std::endl;

    
    delete[] true_labels;
    delete[] pred_labels;

    return 0;
}