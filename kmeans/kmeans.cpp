// ALl the metrics are calculated by the double
// TDIST indicates the type for main computations - 
// while TSTORAGE indicates the data storage type.

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

// using namespace std::chrono
const bool USE_FIXED_SEED = true; // Set to false to use std::random_device

// There are 25 TDIST variables, 6 TSTORAGE variables
template<class TDIST, class TSTORAGE>
class KMeans {
private:
    int numPoints;           // Number of data points
    int numFeatures;         // Number of features (dimensions)
    int k;                   // Number of clusters
    std::vector<TSTORAGE> data; // 1D vector storing data in row-major order
    std::vector<int> labels;  // Cluster assignment for each point
    std::vector<int> groundTruth; // Ground truth labels
    std::vector<TSTORAGE> centroids; // Centroids in 1D (k * numFeatures)
    double runtime;          // Runtime in seconds
    unsigned int seed;       // Random seed for reproducibility
    bool useFixedSeed;       // Flag to determine if a fixed seed is used

    // Helper function to get data point at index
    std::vector<TDIST> getPoint(int idx) const {
        if (idx < 0 || idx >= numPoints) {
            std::cerr << "Error: Invalid point index " << idx << std::endl;
            return std::vector<TDIST>(numFeatures, 0.0);
        }
        std::vector<TDIST> point(numFeatures);
        for (int j = 0; j < numFeatures; ++j) {
            point[j] = data[idx * numFeatures + j];
        }
        return point;
    }

    // Calculate Euclidean distance between two points
    TDIST euclideanDistance(const std::vector<TDIST>& p1, const std::vector<TDIST>& p2) const {
        if (p1.size() != p2.size()) {
            std::cerr << "Error: Mismatched dimensions in euclideanDistance" << std::endl;
            return 0.0;
        }
        TDIST sum = 0.0;
        for (size_t i = 0; i < p1.size(); ++i) {
            sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }
        return std::sqrt(sum);
    }

    // Initialize centroids using k-means++ algorithm
    void initializeCentroids() {
        // Initialize the random number generator with either a fixed seed or std::random_device
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

        // Choose first centroid randomly
        int firstCentroid = dis(gen);
        std::vector<TDIST> centroid = getPoint(firstCentroid);
        for (int j = 0; j < numFeatures; ++j) {
            centroids[j] = static_cast<TDIST>(centroid[j]);
        }

        // Choose remaining centroids using k-means++
        for (int c = 1; c < k; ++c) {
            std::vector<TDIST> distances(numPoints, std::numeric_limits<TDIST>::max());
            for (int i = 0; i < numPoints; ++i) {
                auto point = getPoint(i);
                TDIST minDist = std::numeric_limits<TDIST>::max();
                for (int j = 0; j < c; ++j) {
                    std::vector<TDIST> cent(numFeatures);
                    for (int f = 0; f < numFeatures; ++f) {
                        cent[f] = centroids[j * numFeatures + f];
                    }
                    TDIST dist = static_cast<TDIST>(euclideanDistance(point, cent));
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
    // Constructor with optional seed parameter
    KMeans(int k_, int numFeatures_, unsigned int seed_ = 0, bool useFixedSeed_ = false)
        : k(k_), numFeatures(numFeatures_), numPoints(0), runtime(0.0),
          seed(seed_), useFixedSeed(useFixedSeed_) {}

    // Read data from file (assuming CSV format: feature1,feature2,...,featureN)
    bool readData(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return false;
        }

        std::vector<std::vector<TSTORAGE>> tempData;
        std::string line;
        while (std::getline(file, line)) {
            std::vector<TSTORAGE> point;
            std::stringstream ss(line);
            std::string value;

            while (std::getline(ss, value, ',')) {
                try {
                    point.push_back(std::stod(value));
                } catch (...) {
                    std::cerr << "Error parsing value: " << value << std::endl;
                    return false;
                }
            }

            if (point.size() != static_cast<size_t>(numFeatures)) {
                std::cerr << "Incorrect number of features in row: expected " << numFeatures << ", got " << point.size() << std::endl;
                return false;
            }
            tempData.push_back(point);
        }

        numPoints = tempData.size();
        data.resize(numPoints * numFeatures);
        for (int i = 0; i < numPoints; ++i) {
            for (int j = 0; j < numFeatures; ++j) {
                data[i * numFeatures + j] = tempData[i][j];
            }
        }

        labels.resize(numPoints, 0);
        centroids.resize(k * numFeatures, 0.0);
        file.close();
        return true;
    }

    // Read ground truth labels from file
    bool readGroundTruth(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening ground truth file: " << filename << std::endl;
            return false;
        }

        groundTruth.clear();
        std::string line;
        while (std::getline(file, line)) {
            try {
                int label = std::stoi(line);
                groundTruth.push_back(label);
            } catch (...) {
                std::cerr << "Error parsing ground truth label: " << line << std::endl;
                return false;
            }
        }

        if (groundTruth.size() != static_cast<size_t>(numPoints)) {
            std::cerr << "Number of ground truth labels (" << groundTruth.size() 
                      << ") doesn't match number of points (" << numPoints << ")" << std::endl;
            return false;
        }

        file.close();
        return true;
    }

    // Perform k-means clustering with timing
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
                TDIST minDist = std::numeric_limits<TDIST>::max();
                int newLabel = 0;

                for (int c = 0; c < k; ++c) {
                    std::vector<TDIST> centroid(numFeatures);
                    for (int j = 0; j < numFeatures; ++j) {
                        centroid[j] = centroids[c * numFeatures + j];
                    }
                    TDIST dist = static_cast<TDIST>(euclideanDistance(point, centroid));
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

    // Calculate SSE (Error Sum of Squares)
    double calculateSSE() const {
        // std::cout << "Calculating SSE..." << std::endl;
        double sse = 0.0;
        for (int i = 0; i < numPoints; ++i) {
            auto point = getPoint(i);
            int cluster = labels[i];
            std::vector<TDIST> centroid(numFeatures);
            for (int j = 0; j < numFeatures; ++j) {
                centroid[j] = centroids[cluster * numFeatures + j];
            }
            TDIST dist = euclideanDistance(point, centroid);
            sse += dist * dist;
        }
        return sse;
    }

    // Calculate Adjusted Mutual Information (AMI)
    double calculateAMI() const {
        if (groundTruth.empty()) {
            std::cerr << "No ground truth labels available for AMI calculation" << std::endl;
            return 0.0;
        }

        // std::cout << "Calculating AMI..." << std::endl;

        // Determine the number of unique clusters and labels
        int maxLabel = *std::max_element(groundTruth.begin(), groundTruth.end()) + 1;
        int maxCluster = k;

        // Calculate contingency table
        std::vector<std::vector<int>> contingency(maxCluster, std::vector<int>(maxLabel, 0));
        for (int i = 0; i < numPoints; ++i) {
            contingency[labels[i]][groundTruth[i]]++;
        }

        // Calculate marginal sums
        std::vector<int> a(maxCluster, 0), b(maxLabel, 0);
        for (int i = 0; i < maxCluster; ++i) {
            for (int j = 0; j < maxLabel; ++j) {
                a[i] += contingency[i][j];
                b[j] += contingency[i][j];
            }
        }

        // Calculate Mutual Information (MI)
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
        // std::cout << "Mutual Information (MI): " << mi << std::endl;

        // Calculate entropies
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
        // std::cout << "Entropy of predicted labels (Ha): " << ha << std::endl;
        // std::cout << "Entropy of true labels (Hb): " << hb << std::endl;

        // Calculate expected MI (simplified approximation)
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
        // std::cout << "Expected Mutual Information (EMI): " << emi << std::endl;

        // Calculate AMI
        if (ha + hb == 0) {
            std::cout << "Both entropies are zero, returning AMI = 0.0" << std::endl;
            return 0.0;
        }
        double denominator = (ha + hb) / 2.0 - emi;
        if (denominator == 0) {
            std::cout << "Denominator is zero, returning AMI = 0.0" << std::endl;
            return 0.0;
        }
        double ami = (mi - emi) / denominator;
        // std::cout << "AMI before bounds check: " << ami << std::endl;

        // Ensure AMI is in the valid range [-1, 1]
        ami = std::max(-1.0, std::min(1.0, ami));
        return ami;
    }

    // Calculate Adjusted Rand Index (ARI)
    double calculateARI() const {
        if (groundTruth.empty()) {
            std::cerr << "No ground truth labels available for ARI calculation" << std::endl;
            return 0.0;
        }

        // std::cout << "Calculating ARI..." << std::endl;

        // Calculate contingency table
        int maxLabel = *std::max_element(groundTruth.begin(), groundTruth.end()) + 1;
        int maxCluster = k;
        std::vector<std::vector<int>> contingency(maxCluster, std::vector<int>(maxLabel, 0));
        for (int i = 0; i < numPoints; ++i) {
            contingency[labels[i]][groundTruth[i]]++;
        }

        // Calculate sums
        std::vector<int> a(maxCluster, 0), b(maxLabel, 0);
        for (int i = 0; i < maxCluster; ++i) {
            for (int j = 0; j < maxLabel; ++j) {
                a[i] += contingency[i][j];
                b[j] += contingency[i][j];
            }
        }

        // Calculate indices
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

    // Save labels to CSV file
    bool saveLabelsToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening output labels file: " << filename << std::endl;
            return false;
        }

        // Write header
        file << "point_index,cluster_label\n";
        
        // Write labels
        for (int i = 0; i < numPoints; ++i) {
            file << i << "," << labels[i] << "\n";
        }

        file.close();
        std::cout << "Output labels saved to: " << filename << std::endl;
        return true;
    }

    // Save runtime to CSV file
    bool saveRuntimeToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening runtime file: " << filename << std::endl;
            return false;
        }

        // Write header and runtime
        file << "runtime_seconds\n";
        file << runtime << "\n";

        file.close();
        std::cout << "Runtime saved to: " << filename << std::endl;
        return true;
    }

    // Save centroids to CSV file
    bool saveCentroidsToCSV(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening centroids file: " << filename << std::endl;
            return false;
        }

        // Write header
        file << "centroid_index";
        for (int j = 0; j < numFeatures; ++j) {
            file << ",feature_" << j;
        }
        file << "\n";

        // Write centroids
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
    const std::vector<TSTORAGE>& getCentroids() const { return centroids; }
    double getRuntime() const { return runtime; }
};


int main(int argc, char *argv[]) {
    size_t K(2), NUM_FEATURES(2);
    size_t SEED(42);

    if(argc == 2){
        K = atoi(argv[1]);
    }else if(argc == 3){
        K = atoi(argv[1]);
        NUM_FEATURES = atoi(argv[2]);
    }else if(argc > 3){
        K = atoi(argv[1]);
        NUM_FEATURES = atoi(argv[2]); 
        SEED = atoi(argv[3]); 
    }
    
    // Initialize KMeans with a fixed seed
    KMeans<float, float> kmeans(K, NUM_FEATURES, SEED, USE_FIXED_SEED);

    // Read data and ground truth
    if (!kmeans.readData("../data/blobs/X_2d_10.csv")) {
        return 1;
    }
    if (!kmeans.readGroundTruth("../data/blobs/y_2d_10.csv")) {
        return 1;
    }

    // Perform clustering
    kmeans.fit();

    // Save output labels, runtime, and centroids to CSV files
    if (!kmeans.saveLabelsToCSV("../results/kmeans/output_labels.csv")) {
        return 1;
    }
    if (!kmeans.saveRuntimeToCSV("../results/kmeans/runtime.csv")) {
        return 1;
    }
    if (!kmeans.saveCentroidsToCSV("../results/kmeans/centroids.csv")) {
        return 1;
    }

    // Calculate and print evaluation metrics
    std::cout << "\nStarting evaluation metrics calculation..." << std::endl;
    double sse = kmeans.calculateSSE();
    double ami = kmeans.calculateAMI();
    double ari = kmeans.calculateARI();

    std::cout << "\nEvaluation Metrics:" << std::endl;
    std::cout << "SSE: " << sse << std::endl;
    std::cout << "AMI: " << ami << std::endl;
    std::cout << "ARI: " << ari << std::endl;

    // Print centroids
    std::cout << "\nPrinting centroids..." << std::endl;
    const auto& centroids = kmeans.getCentroids();
    std::cout << "\nCentroids:\n";
    for (int c = 0; c < K; ++c) {
        std::cout << "Centroid " << c << ": ";
        for (int j = 0; j < NUM_FEATURES; ++j) {
            std::cout << centroids[c * NUM_FEATURES + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}




