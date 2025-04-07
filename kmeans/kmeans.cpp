#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <unordered_map>

const bool USE_FIXED_SEED = true;

struct DataPoint {
    std::vector<double> features;
    int label;
};

std::vector<DataPoint> read_csv(const std::string& filename) {
    std::vector<DataPoint> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return data;
    }

    std::string line;
    getline(file, line); 
    
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> features;
        
        // Skip the index column
        getline(ss, value, ',');  // Ignore the first value (index)
        
        while (getline(ss, value, ',')) {
            features.push_back(std::stod(value));
        }

        // Last value is the true label
        int true_label = (int)features.back();
        features.pop_back();
        data.push_back({features, true_label});
    }
    
    std::cout << "Loaded " << data.size() << " data points with "  
              << (data.empty() ? 0 : data[0].features.size()) << " features each" << std::endl;
    
    file.close();
    return data;
}

class KMeans {
private:
    int numPoints;
    int numFeatures;
    int k;
    std::vector<double> data;
    std::vector<int> labels;
    std::vector<int> groundTruth;
    std::vector<double> centroids;
    double runtime;
    unsigned int seed;
    bool useFixedSeed;

    std::vector<double> getPoint(int idx) const {
        if (idx < 0 || idx >= numPoints) {
            std::cerr << "Error: Invalid point index " << idx << std::endl;
            return std::vector<double>(numFeatures, 0.0);
        }
        std::vector<double> point(numFeatures);
        for (int j = 0; j < numFeatures; ++j) {
            point[j] = data[idx * numFeatures + j];
        }
        return point;
    }

    double euclideanDistance(const std::vector<double>& p1, const std::vector<double>& p2) const {
        if (p1.size() != p2.size()) {
            std::cerr << "Error: Mismatched dimensions in euclideanDistance" << std::endl;
            return 0.0;
        }
        double sum = 0.0;
        for (size_t i = 0; i < p1.size(); ++i) {
            sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }
        return std::sqrt(sum);
    }

    void initializeCentroids() {
        std::mt19937 gen;
        if (useFixedSeed) {
            gen = std::mt19937(seed);
            std::cout << "Using fixed seed: " << seed << std::endl;
        } else {
            std::random_device rd;
            gen = std::mt19937(rd());
            std::cout << "Using random_device for seed" << std::endl;
        }

        std::uniform_int_distribution<> dis(0, numPoints - 1);
        int firstCentroid = dis(gen);
        std::vector<double> centroid = getPoint(firstCentroid);
        for (int j = 0; j < numFeatures; ++j) {
            centroids[j] = static_cast<double>(centroid[j]);
        }

        for (int c = 1; c < k; ++c) {
            std::vector<double> distances(numPoints, std::numeric_limits<double>::max());
            for (int i = 0; i < numPoints; ++i) {
                auto point = getPoint(i);
                double minDist = std::numeric_limits<double>::max();
                for (int j = 0; j < c; ++j) {
                    std::vector<double> cent(numFeatures);
                    for (int f = 0; f < numFeatures; ++f) {
                        cent[f] = centroids[j * numFeatures + f];
                    }
                    double dist = static_cast<double>(euclideanDistance(point, cent));
                    minDist = std::min(minDist, dist);
                }
                distances[i] = minDist * minDist;
            }

            std::discrete_distribution<> dist(distances.begin(), distances.end());
            int nextCentroid = dist(gen);
            auto newCentroid = getPoint(nextCentroid);
            for (int j = 0; j < numFeatures; ++j) {
                centroids[c * numFeatures + j] = newCentroid[j];
            }
        }
    }

public:
    KMeans(int k_, int numFeatures_, unsigned int seed_ = 0, bool useFixedSeed_ = false)
        : k(k_), numFeatures(numFeatures_), numPoints(0), runtime(0.0),
          seed(seed_), useFixedSeed(useFixedSeed_) {}

    bool loadFromDataPoints(const std::vector<DataPoint>& dataPoints) {
        if (dataPoints.empty()) {
            std::cerr << "No data points provided" << std::endl;
            return false;
        }

        numPoints = dataPoints.size();
        if (dataPoints[0].features.size() != static_cast<size_t>(numFeatures)) {
            std::cerr << "Feature count mismatch: expected " << numFeatures 
                      << ", got " << dataPoints[0].features.size() << std::endl;
            numPoints = 0;
            return false;
        }

        data.resize(numPoints * numFeatures);
        groundTruth.resize(numPoints);
        labels.resize(numPoints, 0);
        centroids.resize(k * numFeatures, 0.0);

        for (int i = 0; i < numPoints; ++i) {
            for (int j = 0; j < numFeatures; ++j) {
                data[i * numFeatures + j] = static_cast<double>(dataPoints[i].features[j]);
            }
            groundTruth[i] = dataPoints[i].label;
        }
        return true;
    }

    void fit(int maxIterations = 100) {
        if (numPoints < k) {
            std::cerr << "Number of points (" << numPoints << ") less than k (" << k << ")" << std::endl;
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();
        initializeCentroids();
        bool changed = true;
        int iterations = 0;

        while (changed && iterations < maxIterations) {
            changed = false;

            for (int i = 0; i < numPoints; ++i) {
                auto point = getPoint(i);
                double minDist = std::numeric_limits<double>::max();
                int newLabel = 0;

                for (int c = 0; c < k; ++c) {
                    std::vector<double> centroid(numFeatures);
                    for (int j = 0; j < numFeatures; ++j) {
                        centroid[j] = centroids[c * numFeatures + j];
                    }
                    double dist = static_cast<double>(euclideanDistance(point, centroid));
                    if (dist < minDist) {
                        minDist = dist;
                        newLabel = c;
                    }
                }

                if (labels[i] != newLabel) {
                    labels[i] = newLabel;
                    changed = true;
                }
            }

            std::vector<int> counts(k, 0);
            std::fill(centroids.begin(), centroids.end(), 0.0);

            for (int i = 0; i < numPoints; ++i) {
                int cluster = labels[i];
                counts[cluster]++;
                for (int j = 0; j < numFeatures; ++j) {
                    centroids[cluster * numFeatures + j] += data[i * numFeatures + j];
                }
            }

            for (int c = 0; c < k; ++c) {
                if (counts[c] > 0) {
                    for (int j = 0; j < numFeatures; ++j) {
                        centroids[c * numFeatures + j] /= counts[c];
                    }
                }
            }

            iterations++;
        }

        auto end = std::chrono::high_resolution_clock::now();
        runtime = std::chrono::duration<double>(end - start).count();

        std::cout << "Converged after " << iterations << " iterations" << std::endl;
        std::cout << "Runtime: " << runtime << " seconds" << std::endl;
    }

    double calculateSSE() const {
        double sse = 0.0;
        for (int i = 0; i < numPoints; ++i) {
            auto point = getPoint(i);
            int cluster = labels[i];
            std::vector<double> centroid(numFeatures);
            for (int j = 0; j < numFeatures; ++j) {
                centroid[j] = centroids[cluster * numFeatures + j];
            }
            double dist = euclideanDistance(point, centroid);
            sse += dist * dist;
        }
        return sse;
    }

    double calculateAMI() const {
        if (groundTruth.empty()) {
            std::cerr << "No ground truth labels available for AMI calculation" << std::endl;
            return 0.0;
        }

        int maxLabel = *std::max_element(groundTruth.begin(), groundTruth.end()) + 1;
        int maxCluster = k;
        std::vector<std::vector<int>> contingency(maxCluster, std::vector<int>(maxLabel, 0));
        for (int i = 0; i < numPoints; ++i) {
            contingency[labels[i]][groundTruth[i]]++;
        }

        std::vector<int> a(maxCluster, 0), b(maxLabel, 0);
        for (int i = 0; i < maxCluster; ++i) {
            for (int j = 0; j < maxLabel; ++j) {
                a[i] += contingency[i][j];
                b[j] += contingency[i][j];
            }
        }

        double mi = 0.0;
        for (int i = 0; i < maxCluster; ++i) {
            for (int j = 0; j < maxLabel; ++j) {
                if (contingency[i][j] > 0) {
                    double p_ij = contingency[i][j] / static_cast<double>(numPoints);
                    double p_i = a[i] / static_cast<double>(numPoints);
                    double p_j = b[j] / static_cast<double>(numPoints);
                    mi += p_ij * std::log(p_ij / (p_i * p_j));
                }
            }
        }

        double ha = 0.0, hb = 0.0;
        for (int i = 0; i < maxCluster; ++i) {
            if (a[i] > 0) {
                double p_i = a[i] / static_cast<double>(numPoints);
                ha -= p_i * std::log(p_i);
            }
        }
        for (int j = 0; j < maxLabel; ++j) {
            if (b[j] > 0) {
                double p_j = b[j] / static_cast<double>(numPoints);
                hb -= p_j * std::log(p_j);
            }
        }

        double emi = 0.0;
        for (int i = 0; i < maxCluster; ++i) {
            for (int j = 0; j < maxLabel; ++j) {
                if (a[i] > 0 && b[j] > 0) {
                    double nij = (static_cast<double>(a[i]) * b[j]) / numPoints;
                    if (nij > 0) {
                        emi += (nij / numPoints) * std::log(numPoints * nij / (a[i] * b[j]));
                    }
                }
            }
        }

        if (ha + hb == 0) return 0.0;
        double denominator = (ha + hb) / 2.0 - emi;
        if (denominator == 0) return 0.0;
        double ami = (mi - emi) / denominator;
        return std::max(-1.0, std::min(1.0, ami));
    }

    double calculateARI() const {
        if (groundTruth.empty()) {
            std::cerr << "No ground truth labels available for ARI calculation" << std::endl;
            return 0.0;
        }

        int maxLabel = *std::max_element(groundTruth.begin(), groundTruth.end()) + 1;
        int maxCluster = k;
        std::vector<std::vector<int>> contingency(maxCluster, std::vector<int>(maxLabel, 0));
        for (int i = 0; i < numPoints; ++i) {
            contingency[labels[i]][groundTruth[i]]++;
        }

        std::vector<int> a(maxCluster, 0), b(maxLabel, 0);
        for (int i = 0; i < maxCluster; ++i) {
            for (int j = 0; j < maxLabel; ++j) {
                a[i] += contingency[i][j];
                b[j] += contingency[i][j];
            }
        }

        double sum_nij = 0.0, sum_a = 0.0, sum_b = 0.0;
        for (int i = 0; i < maxCluster; ++i) {
            for (int j = 0; j < maxLabel; ++j) {
                sum_nij += (static_cast<double>(contingency[i][j]) * (contingency[i][j] - 1)) / 2.0;
            }
            sum_a += (static_cast<double>(a[i]) * (a[i] - 1)) / 2.0;
        }
        for (int j = 0; j < maxLabel; ++j) {
            sum_b += (static_cast<double>(b[j]) * (b[j] - 1)) / 2.0;
        }

        double n = numPoints;
        double expected = (sum_a * sum_b) / (n * (n - 1) / 2.0);
        double max_index = (sum_a + sum_b) / 2.0;
        double index = sum_nij;

        if (max_index == expected) return 0.0;
        return (index - expected) / (max_index - expected);
    }

    bool saveLabelsToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening output labels file: " << filename << std::endl;
            return false;
        }

        file << "point_index,cluster_label\n";
        for (int i = 0; i < numPoints; ++i) {
            file << i << "," << labels[i] << "\n";
        }

        file.close();
        std::cout << "Output labels saved to: " << filename << std::endl;
        return true;
    }

    bool saveRuntimeToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening runtime file: " << filename << std::endl;
            return false;
        }

        file << "runtime_seconds\n" << runtime << "\n";
        file.close();
        std::cout << "Runtime saved to: " << filename << std::endl;
        return true;
    }

    bool saveCentroidsToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening centroids file: " << filename << std::endl;
            return false;
        }

        file << "centroid_index";
        for (int j = 0; j < numFeatures; ++j) {
            file << ",feature_" << j;
        }
        file << "\n";

        for (int c = 0; c < k; ++c) {
            file << c;
            for (int j = 0; j < numFeatures; ++j) {
                file << "," << centroids[c * numFeatures + j];
            }
            file << "\n";
        }

        file.close();
        std::cout << "Centroids saved to: " << filename << std::endl;
        return true;
    }

    const std::vector<int>& getLabels() const { return labels; }
    const std::vector<double>& getCentroids() const { return centroids; }
    double getRuntime() const { return runtime; }
};

int main(int argc, char *argv[]) {
    size_t K(2), NUM_FEATURES(2);
    size_t SEED(42);

    if(argc == 2){
        K = atoi(argv[1]);
    } else if(argc == 3){
        K = atoi(argv[1]);
        NUM_FEATURES = atoi(argv[2]);
    } else if(argc > 3){
        K = atoi(argv[1]);
        NUM_FEATURES = atoi(argv[2]); 
        SEED = atoi(argv[3]); 
    }
    
    KMeans kmeans(K, NUM_FEATURES, SEED, USE_FIXED_SEED);

    std::vector<DataPoint> dataPoints = read_csv("../data/clustering/blobs_20d_10_include_y.csv");
    if (dataPoints.empty()) {
        std::cerr << "Failed to read CSV data" << std::endl;
        return 1;
    }

    if (!kmeans.loadFromDataPoints(dataPoints)) {
        std::cerr << "Failed to load data points into KMeans" << std::endl;
        return 1;
    }

    kmeans.fit();
    kmeans.saveLabelsToCSV("../results/kmeans/output_labels.csv");
    kmeans.saveRuntimeToCSV("../results/kmeans/runtime.csv");
    kmeans.saveCentroidsToCSV("../results/kmeans/centroids.csv");

    double SSE = kmeans.calculateSSE(); 
    std::cout << "\nEvaluation Metrics:" << std::endl;
    std::cout << "SSE: " << SSE << std::endl;
    std::cout << "AMI: " << kmeans.calculateAMI() << std::endl;
    std::cout << "ARI: " << kmeans.calculateARI() << std::endl;

    // const auto& centroids = kmeans.getCentroids();
    // std::cout << "\nCentroids:\n";
    // for (int c = 0; c < K; ++c) {
    //     std::cout << "Centroid " << c << ": ";
    //     for (int j = 0; j < NUM_FEATURES; ++j) {
    //         std::cout << centroids[c * NUM_FEATURES + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}