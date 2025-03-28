#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

struct DataPoint {
    std::vector<double> features;
    int true_label;  // Ground truth
    int cluster;     // Predicted cluster
};

class DBSCAN {
private:
    double eps;
    int min_pts;
    std::vector<DataPoint> data;

    double distance(const std::vector<double>& p1, const std::vector<double>& p2) {
        double sum = 0.0;
        for (size_t i = 0; i < p1.size(); ++i) {
            double diff = p1[i] - p2[i];
            sum += diff * diff;
        }
        return sqrt(sum);
    }

    std::vector<int> find_neighbors(int point_idx) {
        std::vector<int> neighbors;
        for (size_t i = 0; i < data.size(); ++i) {
            if (i != point_idx && distance(data[point_idx].features, data[i].features) <= eps) {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    }

    void expand_cluster(int point_idx, int cluster_label, std::unordered_set<int>& visited) {
        data[point_idx].cluster = cluster_label;
        std::vector<int> neighbors = find_neighbors(point_idx);
        
        if (neighbors.size() < min_pts) return;
        
        for (int neighbor_idx : neighbors) {
            if (visited.find(neighbor_idx) == visited.end()) {
                visited.insert(neighbor_idx);
                if (data[neighbor_idx].cluster == -1) {
                    expand_cluster(neighbor_idx, cluster_label, visited);
                }
            }
        }
    }

public:
    DBSCAN(double epsilon = 0.01, int min_points = 5) : eps(epsilon), min_pts(min_points) {}
    
    void fit(const std::vector<DataPoint>& input_data) {
        data = input_data;
        for (auto& point : data) {
            point.cluster = -1;
        }
        
        int cluster_label = 0;
        std::unordered_set<int> visited;
        
        for (size_t i = 0; i < data.size(); ++i) {
            if (visited.find(i) != visited.end()) continue;
            
            visited.insert(i);
            std::vector<int> neighbors = find_neighbors(i);
            
            if (neighbors.size() >= min_pts) {
                expand_cluster(i, cluster_label, visited);
                cluster_label++;
            }
        }
    }
    
    std::vector<int> get_labels() const {
        std::vector<int> labels;
        for (const auto& point : data) {
            labels.push_back(point.cluster);
        }
        return labels;
    }
};

double adjusted_mutual_information(const std::vector<int>& true_labels, 
                                 const std::vector<int>& pred_labels) {
    int n = true_labels.size();
    if (n == 0) return 0.0;

    std::unordered_map<int, std::unordered_map<int, int>> contingency;
    
    // Build contingency table
    for (int i = 0; i < n; ++i) {
        contingency[true_labels[i]][pred_labels[i]]++;
    }
    
    // Calculate marginal sums
    std::unordered_map<int, int> a_sums, b_sums;
    for (const auto& row : contingency) {
        for (const auto& cell : row.second) {
            a_sums[row.first] += cell.second;
            b_sums[cell.first] += cell.second;
        }
    }
    
    // Mutual Information (MI)
    double mi = 0.0;
    double log_n = log(n);
    for (const auto& row : contingency) {
        for (const auto& cell : row.second) {
            double nij = cell.second;
            if (nij == 0) continue;
            double ai = a_sums[row.first];
            double bj = b_sums[cell.first];
            double denom = ai * bj;
            if (denom > 0) {  // Avoid log(0) or division by 0
                mi += nij * log(nij * n / denom + 1e-10);  // Add small epsilon
            }
        }
    }
    mi = (mi > 0 ? mi / (n * log(2.0)) : 0);  // Convert to base 2

    // Entropy of true labels (H(A))
    double h_true = 0.0;
    for (const auto& a : a_sums) {
        double p = a.second / (double)n;
        if (p > 0) h_true -= p * log(p);
    }
    h_true /= log(2.0);

    // Entropy of predicted labels (H(B))
    double h_pred = 0.0;
    for (const auto& b : b_sums) {
        double p = b.second / (double)n;
        if (p > 0) h_pred -= p * log(p);
    }
    h_pred /= log(2.0);

    // Expected MI (simplified approximation)
    double emi = 0.0;  // Could compute more precisely, but often small
    double max_mi = (h_true + h_pred) / 2.0;

    // AMI calculation with protection against NaN
    double denom = max_mi - emi + 1e-10;
    if (denom <= 0) return 0.0;
    return (mi - emi) / denom;
}

// Adjusted Rand Index (unchanged, as it typically works fine)
double adjusted_rand_index(const std::vector<int>& true_labels, 
                         const std::vector<int>& pred_labels) {
    int n = true_labels.size();
    std::unordered_map<int, std::unordered_map<int, int>> contingency;
    
    for (int i = 0; i < n; ++i) {
        contingency[true_labels[i]][pred_labels[i]]++;
    }
    
    double sum_nij2 = 0.0;
    std::unordered_map<int, double> a_sums, b_sums;
    for (const auto& row : contingency) {
        for (const auto& cell : row.second) {
            sum_nij2 += (cell.second * (cell.second - 1)) / 2.0;
            a_sums[row.first] += cell.second;
            b_sums[cell.first] += cell.second;
        }
    }
    
    double sum_ai2 = 0.0, sum_bj2 = 0.0;
    for (const auto& a : a_sums) sum_ai2 += (a.second * (a.second - 1)) / 2.0;
    for (const auto& b : b_sums) sum_bj2 += (b.second * (b.second - 1)) / 2.0;
    
    double n_choose_2 = (n * (n - 1)) / 2.0;
    double expected = (sum_ai2 * sum_bj2) / n_choose_2;
    double max_index = (sum_ai2 + sum_bj2) / 2.0;
    double index = sum_nij2;
    
    return (index - expected) / (max_index - expected + 1e-10);
}

std::vector<DataPoint> scale_features(const std::vector<DataPoint>& data) {
    std::vector<DataPoint> scaled_data = data;
    if (data.empty()) return scaled_data;
    
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

// Modified CSV Reader for new format
std::vector<DataPoint> read_csv(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    getline(file, line);  // Skip header: ,feature1,feature2,...,label
    
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        
        // Skip the index column
        getline(ss, value, ',');  // Ignore the first value (index)
        
        // Read features
        while (getline(ss, value, ',')) {
            features.push_back(std::stod(value));
        }
        
        // Last value is the true label
        int true_label = (int)features.back();
        features.pop_back();
        data.push_back({features, true_label, -1});
    }
    file.close();
    return data;
}

void write_predictions(const std::vector<DataPoint>& data, 
                      const std::vector<int>& labels, 
                      const std::string& filename) {
    std::ofstream file(filename);
    file << ",feature1,feature2,...,true_label,cluster\n";  // Match input format
    
    for (size_t i = 0; i < data.size(); ++i) {
        file << i + 1 << ",";  // Write index
        for (size_t j = 0; j < data[i].features.size(); ++j) {
            file << data[i].features[j] << ",";
        }
        file << data[i].true_label << "," << labels[i] << "\n";
    }
    file.close();
}

int main() {
    std::vector<DataPoint> raw_data = read_csv("../data/clustering/shape_clusters_include_y.csv");
    if (raw_data.empty()) {
        std::cerr << "No data loaded. Exiting." << std::endl;
        return 1;
    }
    
    std::vector<DataPoint> data = scale_features(raw_data);
    
    DBSCAN dbscan(0.25, 20);
    auto start = std::chrono::high_resolution_clock::now();
    dbscan.fit(data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Clustering time: " << duration.count() << " ms" << std::endl;
    
    std::vector<int> pred_labels = dbscan.get_labels();
    std::vector<int> true_labels;
    for (const auto& point : data) {
        true_labels.push_back(point.true_label);
    }
    
    int n_clusters = 0;
    for (int label : pred_labels) {
        if (label >= 0) n_clusters = std::max(n_clusters, label + 1);
    }
    std::cout << "Number of clusters found: " << n_clusters << std::endl;
    std::cout << "Number of noise points: " << 
                 std::count(pred_labels.begin(), pred_labels.end(), -1) << std::endl;
    
    double ami = adjusted_mutual_information(true_labels, pred_labels);
    double ari = adjusted_rand_index(true_labels, pred_labels);
    std::cout << "Adjusted Mutual Information (AMI): " << ami << std::endl;
    std::cout << "Adjusted Rand Index (ARI): " << ari << std::endl;
    
    write_predictions(data, pred_labels, "../results/dbscan/clusters.csv");
    
    return 0;
}