#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>

// Simple matrix class for 2D operations
class Matrix {
private:
    std::vector<double> data;
    int rows, cols;

public:
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(r * c, 0.0);
    }

    double& operator()(int i, int j) { return data[i * cols + j]; }
    double operator()(int i, int j) const { return data[i * cols + j]; }
    int get_rows() const { return rows; }
    int get_cols() const { return cols; }
};

// Vector operations
double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double norm(const std::vector<double>& v) {
    return std::sqrt(dot(v, v));
}

std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) result[i] = a[i] - b[i];
    return result;
}

struct DataPoint {
    std::vector<double> features;
    int label;  // -1 if no label provided
};

class GMM {
private:
    int n_features;
    int n_components;
    std::vector<double> weights;           // pi_k
    std::vector<std::vector<double>> means; // mu_k
    std::vector<Matrix> covariances;       // Sigma_k (diagonal for simplicity)
    double log_likelihood;

    double gaussian_pdf(const std::vector<double>& x, const std::vector<double>& mean, 
                       const Matrix& cov) {
        int d = x.size();
        std::vector<double> diff = subtract(x, mean);
        double det = 1.0;
        for (int i = 0; i < d; ++i) det *= cov(i, i);  // Determinant of diagonal cov

        double exponent = 0.0;
        for (int i = 0; i < d; ++i) {
            exponent += diff[i] * diff[i] / cov(i, i);
        }
        return std::exp(-0.5 * exponent) / std::sqrt(std::pow(2 * M_PI, d) * det);
    }

public:
    GMM(int k) : n_components(k), log_likelihood(0.0) {}

    void initialize(const std::vector<DataPoint>& data) {
        n_features = data[0].features.size();
        weights.resize(n_components, 1.0 / n_components);
        means.resize(n_components, std::vector<double>(n_features));
        covariances.resize(n_components, Matrix(n_features, n_features));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, data.size() - 1);

        // Randomly initialize means from data points
        std::vector<int> used;
        for (int k = 0; k < n_components; ++k) {
            int idx;
            do { idx = dis(gen); } while (std::find(used.begin(), used.end(), idx) != used.end());
            used.push_back(idx);
            means[k] = data[idx].features;
        }

        // Initialize covariances as identity matrices
        for (int k = 0; k < n_components; ++k) {
            for (int i = 0; i < n_features; ++i) {
                covariances[k](i, i) = 1.0;
            }
        }
    }

    double fit(const std::vector<DataPoint>& data, int max_iter = 100, double tol = 1e-4) {
        initialize(data);
        int n = data.size();
        Matrix responsibilities(n, n_components);

        for (int iter = 0; iter < max_iter; ++iter) {
            // E-step: Compute responsibilities
            double new_log_likelihood = 0.0;
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int k = 0; k < n_components; ++k) {
                    responsibilities(i, k) = weights[k] * gaussian_pdf(data[i].features, means[k], covariances[k]);
                    sum += responsibilities(i, k);
                }
                for (int k = 0; k < n_components; ++k) {
                    responsibilities(i, k) /= sum;
                }
                new_log_likelihood += std::log(sum);
            }

            // Check convergence
            if (iter > 0 && std::abs(new_log_likelihood - log_likelihood) < tol) break;
            log_likelihood = new_log_likelihood;

            // M-step: Update parameters
            std::vector<double> Nk(n_components, 0.0);
            for (int k = 0; k < n_components; ++k) {
                Nk[k] = 0.0;
                for (int i = 0; i < n; ++i) {
                    Nk[k] += responsibilities(i, k);
                }
                weights[k] = Nk[k] / n;

                // Update mean
                means[k].assign(n_features, 0.0);
                for (int i = 0; i < n; ++i) {
                    for (int d = 0; d < n_features; ++d) {
                        means[k][d] += responsibilities(i, k) * data[i].features[d];
                    }
                }
                for (int d = 0; d < n_features; ++d) {
                    means[k][d] /= Nk[k];
                }

                // Update covariance (diagonal)
                for (int d = 0; d < n_features; ++d) {
                    covariances[k](d, d) = 0.0;
                    for (int i = 0; i < n; ++i) {
                        double diff = data[i].features[d] - means[k][d];
                        covariances[k](d, d) += responsibilities(i, k) * diff * diff;
                    }
                    covariances[k](d, d) = std::max(covariances[k](d, d) / Nk[k], 1e-6); // Avoid singularity
                }
            }
        }
        return log_likelihood;
    }

    std::vector<int> predict(const std::vector<DataPoint>& data) {
        std::vector<int> labels(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            double max_prob = -1.0;
            int best_k = 0;
            for (int k = 0; k < n_components; ++k) {
                double prob = weights[k] * gaussian_pdf(data[i].features, means[k], covariances[k]);
                if (prob > max_prob) {
                    max_prob = prob;
                    best_k = k;
                }
            }
            labels[i] = best_k;
        }
        return labels;
    }

    std::vector<std::vector<double>> predict_proba(const std::vector<DataPoint>& data) {
        std::vector<std::vector<double>> probs(data.size(), std::vector<double>(n_components));
        for (size_t i = 0; i < data.size(); ++i) {
            double sum = 0.0;
            for (int k = 0; k < n_components; ++k) {
                probs[i][k] = weights[k] * gaussian_pdf(data[i].features, means[k], covariances[k]);
                sum += probs[i][k];
            }
            for (int k = 0; k < n_components; ++k) {
                probs[i][k] /= sum;
            }
        }
        return probs;
    }
};

std::vector<DataPoint> read_csv(const std::string& filename, int& n_features) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return data;
    }
    std::string line;
    if (!getline(file, line)) return data;
    std::cout << "Header: " << line << std::endl;

    std::stringstream ss(line);
    std::string value;
    int cols = 0;
    while (getline(ss, value, ',')) cols++;
    n_features = cols - 2;  // Exclude index and label

    int line_num = 1;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        int label = -1;
        int column = 0;

        while (getline(ss, value, ',')) {
            if (column == 0) {  // Skip index
                column++;
                continue;
            }
            try {
                if (column < cols - 1) {  // Features
                    features.push_back(std::stod(value));
                } else {  // Label
                    label = std::stoi(value);
                }
            } catch (const std::exception& e) {
                if (column == cols - 1) label = -1;  // No label provided
                else {
                    std::cerr << "Error parsing '" << value << "' at line " << line_num 
                              << ", column " << column << std::endl;
                    return {};
                }
            }
            column++;
        }
        if (features.size() != n_features) {
            std::cerr << "Error: Expected " << n_features << " features, got " 
                      << features.size() << " at line " << line_num << std::endl;
            return {};
        }
        data.push_back({features, label});
        line_num++;
    }
    std::cout << "Loaded " << data.size() << " data points with " 
              << n_features << " features each" << std::endl;
    return data;
}

double compute_accuracy(const std::vector<DataPoint>& data, const std::vector<int>& labels) {
    int n = data.size();
    std::vector<int> true_labels(n);
    bool has_labels = false;
    for (int i = 0; i < n; ++i) {
        true_labels[i] = data[i].label;
        if (true_labels[i] != -1) has_labels = true;
    }
    if (!has_labels) return -1.0;

    // Simple accuracy: max matching after permutation (for 2 clusters as example)
    int correct = 0;
    for (int i = 0; i < n; ++i) {
        if (true_labels[i] == labels[i]) correct++;
    }
    int flipped = 0;
    for (int i = 0; i < n; ++i) {
        if (true_labels[i] != labels[i]) flipped++;
    }
    return std::max(correct, flipped) / static_cast<double>(n);
}

void write_results(const std::vector<DataPoint>& data, const std::vector<int>& labels, 
                  const std::vector<std::vector<double>>& probs, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file" << std::endl;
        return;
    }
    file << "feature1,feature2,...,label,predicted_label";
    for (int k = 0; k < probs[0].size(); ++k) {
        file << ",prob_cluster" << k;
    }
    file << "\n";

    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[i].features.size(); ++j) {
            file << data[i].features[j] << ",";
        }
        file << data[i].label << "," << labels[i];
        for (int k = 0; k < probs[i].size(); ++k) {
            file << "," << probs[i][k];
        }
        file << "\n";
    }
}

int main() {
    int n_features;
    std::vector<DataPoint> data = read_csv("../data/clustering/X_20d_10_include_y.csv", n_features);
    if (data.empty()) return 1;

    int n_components = 10;  // Number of clusters
    GMM gmm(n_components);

    auto start = std::chrono::high_resolution_clock::now();
    double log_likelihood = gmm.fit(data);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::vector<int> labels = gmm.predict(data);
    std::vector<std::vector<double>> probs = gmm.predict_proba(data);

    double accuracy = compute_accuracy(data, labels);

    std::cout << "Number of data points: " << data.size() << std::endl;
    std::cout << "Number of features: " << n_features << std::endl;
    std::cout << "Number of clusters: " << n_components << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final log-likelihood: " << log_likelihood << std::endl;
    if (accuracy >= 0.0) {
        std::cout << "Clustering accuracy: " << accuracy << std::endl;
    } else {
        std::cout << "Accuracy: N/A (no true labels provided)" << std::endl;
    }

    write_results(data, labels, probs, "gmm_output.csv");

    return 0;
}