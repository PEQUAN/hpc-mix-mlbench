#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <unordered_map>

struct DataPoint {
    double* features; 
    int label;
    size_t numFeatures;

    DataPoint() : features(nullptr), label(0), numFeatures(0) {}

    DataPoint(size_t numFeatures_) : numFeatures(numFeatures_), label(0) {
        features = new double[numFeatures];
    }

    ~DataPoint() {
        delete[] features;
    }

    DataPoint(const DataPoint& other) : numFeatures(other.numFeatures), label(other.label) {
        features = new double[numFeatures];
        for (size_t i = 0; i < numFeatures; ++i) {
            features[i] = other.features[i];
        }
    }

    DataPoint& operator=(const DataPoint& other) {
        if (this != &other) {
            delete[] features;
            numFeatures = other.numFeatures;
            label = other.label;
            features = new double[numFeatures];
            for (size_t i = 0; i < numFeatures; ++i) {
                features[i] = other.features[i];
            }
        }
        return *this;
    }
};

DataPoint* read_csv(const std::string& filename, size_t numFeatures, size_t& numPoints) {
    std::ifstream file(filename);
    numPoints = 0;
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    std::string line;
    getline(file, line);
    while (getline(file, line)) {
        ++numPoints;
    }
    file.clear();
    file.seekg(0);

    DataPoint* data = new DataPoint[numPoints];
    size_t idx = 0;

    getline(file, line); 
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        getline(ss, value, ',');

        data[idx].numFeatures = numFeatures;
        data[idx].features = new double[numFeatures];
        data[idx].label = 0;

        size_t featureIdx = 0;
        while (getline(ss, value, ',')) {
            if (featureIdx < numFeatures) {
                data[idx].features[featureIdx] = std::stod(value);
            } else {
                data[idx].label = static_cast<int>(std::stod(value));
            }
            ++featureIdx;
        }
        ++idx;
    }

    std::cout << "Loaded " << numPoints << " data points with " << numFeatures << " features each" << std::endl;

    file.close();
    return data;
}

class KMeans {
private:
    int numPoints;
    int numFeatures;
    int k;
    double* data;          // numPoints * numFeatures
    int* labels;           // numPoints
    int* groundTruth;      // numPoints
    double* centroids;     // k * numFeatures
    double runtime;

    void getPoint(int idx, double* point) const {
        if (idx < 0 || idx >= numPoints) {
            std::cerr << "Error: Invalid point index " << idx << std::endl;
            for (int j = 0; j < numFeatures; ++j) {
                point[j] = 0.0;
            }
            return;
        }
        for (int j = 0; j < numFeatures; ++j) {
            point[j] = data[idx * numFeatures + j];
        }
    }

    double euclideanDistance(const double* p1, const double* p2) const {
        double sum = 0.0;
        for (int i = 0; i < numFeatures; ++i) {
            sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }
        return std::sqrt(sum);
    }

    void initializeCentroids() {
        // Deterministically select the first k points as initial centroids
        for (int c = 0; c < k; ++c) {
            if (c >= numPoints) {
                std::cerr << "Not enough points to select " << k << " centroids" << std::endl;
                return;
            }
            double* tempPoint = new double[numFeatures];
            getPoint(c, tempPoint);
            for (int j = 0; j < numFeatures; ++j) {
                centroids[c * numFeatures + j] = tempPoint[j];
            }
            delete[] tempPoint;
        }
        std::cout << "Initialized " << k << " centroids deterministically using first " << k << " points" << std::endl;
    }

public:
    KMeans(int k_, int numFeatures_)
        : k(k_), numFeatures(numFeatures_), numPoints(0), runtime(0.0),
          data(nullptr), labels(nullptr), groundTruth(nullptr), centroids(nullptr) {}

    ~KMeans() {
        delete[] data;
        delete[] labels;
        delete[] groundTruth;
        delete[] centroids;
    }

    bool loadFromDataPoints(const DataPoint* dataPoints, int numPoints_) {
        if (numPoints_ == 0) {
            std::cerr << "No data points provided" << std::endl;
            return false;
        }

        numPoints = numPoints_;
        delete[] data;
        delete[] groundTruth;
        delete[] labels;
        delete[] centroids;

        data = new double[numPoints * numFeatures];
        groundTruth = new int[numPoints];
        labels = new int[numPoints];
        centroids = new double[k * numFeatures];

        for (int i = 0; i < numPoints; ++i) {
            if (static_cast<size_t>(numFeatures) != dataPoints[i].numFeatures) {
                std::cerr << "Feature count mismatch: expected " << numFeatures
                          << ", got " << dataPoints[i].numFeatures << std::endl;
                numPoints = 0;
                delete[] data;
                delete[] groundTruth;
                delete[] labels;
                delete[] centroids;
                data = nullptr;
                groundTruth = nullptr;
                labels = nullptr;
                centroids = nullptr;
                return false;
            }
            for (int j = 0; j < numFeatures; ++j) {
                data[i * numFeatures + j] = dataPoints[i].features[j];
            }
            groundTruth[i] = dataPoints[i].label;
            labels[i] = 0;
        }
        for (int i = 0; i < k * numFeatures; ++i) {
            centroids[i] = 0.0;
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

        double* point = new double[numFeatures];
        double* centroid = new double[numFeatures];

        while (changed && iterations < maxIterations) {
            changed = false;

            for (int i = 0; i < numPoints; ++i) {
                getPoint(i, point);
                double minDist = std::numeric_limits<double>::max();
                int newLabel = 0;

                for (int c = 0; c < k; ++c) {
                    for (int j = 0; j < numFeatures; ++j) {
                        centroid[j] = centroids[c * numFeatures + j];
                    }
                    double dist = euclideanDistance(point, centroid);
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

            int* counts = new int[k]();
            for (int i = 0; i < k * numFeatures; ++i) {
                centroids[i] = 0.0;
            }

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

            delete[] counts;
            iterations++;
        }

        delete[] point;
        delete[] centroid;

        auto end = std::chrono::high_resolution_clock::now();
        runtime = std::chrono::duration<double>(end - start).count();

        std::cout << "Converged after " << iterations << " iterations" << std::endl;
        std::cout << "Runtime: " << runtime << " seconds" << std::endl;
    }

    double calculateSSE() const {
        double sse = 0.0;
        double* point = new double[numFeatures];
        double* centroid = new double[numFeatures];

        for (int i = 0; i < numPoints; ++i) {
            getPoint(i, point);
            int cluster = labels[i];
            for (int j = 0; j < numFeatures; ++j) {
                centroid[j] = centroids[cluster * numFeatures + j];
            }
            double dist = euclideanDistance(point, centroid);
            sse += dist * dist;
        }

        delete[] point;
        delete[] centroid;
        return sse;
    }

    double calculateAMI() const {
        if (!groundTruth) {
            std::cerr << "No ground truth labels available for AMI calculation" << std::endl;
            return 0.0;
        }

        int maxLabel = 0;
        for (int i = 0; i < numPoints; ++i) {
            maxLabel = std::max(maxLabel, groundTruth[i]);
        }
        maxLabel += 1;
        int maxCluster = k;

        int** contingency = new int*[maxCluster];
        for (int i = 0; i < maxCluster; ++i) {
            contingency[i] = new int[maxLabel]();
        }

        for (int i = 0; i < numPoints; ++i) {
            contingency[labels[i]][groundTruth[i]]++;
        }

        int* a = new int[maxCluster]();
        int* b = new int[maxLabel]();
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

        for (int i = 0; i < maxCluster; ++i) {
            delete[] contingency[i];
        }
        delete[] contingency;
        delete[] a;
        delete[] b;

        if (ha + hb == 0) return 0.0;
        double denominator = (ha + hb) / 2.0 - emi;
        if (denominator == 0) return 0.0;
        double ami = (mi - emi) / denominator;
        return std::max(-1.0, std::min(1.0, ami));
    }

    double calculateARI() const {
        if (!groundTruth) {
            std::cerr << "No ground truth labels available for ARI calculation" << std::endl;
            return 0.0;
        }

        int maxLabel = 0;
        for (int i = 0; i < numPoints; ++i) {
            maxLabel = std::max(maxLabel, groundTruth[i]);
        }
        maxLabel += 1;
        int maxCluster = k;

        int** contingency = new int*[maxCluster];
        for (int i = 0; i < maxCluster; ++i) {
            contingency[i] = new int[maxLabel]();
        }

        for (int i = 0; i < numPoints; ++i) {
            contingency[labels[i]][groundTruth[i]]++;
        }

        int* a = new int[maxCluster]();
        int* b = new int[maxLabel]();
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

        for (int i = 0; i < maxCluster; ++i) {
            delete[] contingency[i];
        }
        delete[] contingency;
        delete[] a;
        delete[] b;

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

    const int* getLabels() const { return labels; }
    double* getCentroids() const { return centroids; }
    double getRuntime() const { return runtime; }
};

int main(int argc, char* argv[]) {
    size_t K = 2;
    size_t NUM_FEATURES = 2;

    if (argc == 2) {
        K = atoi(argv[1]);
    } else if (argc >= 3) {
        K = atoi(argv[1]);
        NUM_FEATURES = atoi(argv[2]);
    }

    KMeans kmeans(K, NUM_FEATURES);
/*
From 
python3 blobs.py 10000 2 10
python3 blobs.py 10000 20 10
python3 blobs.py 10000 2 20
python3 blobs.py 10000 2 30

blobs_{dim}d_{n_clusters}_include_y.csv
./kmeans 10 20 - #cluster=10, 20-dimensional data points
*/
    size_t numPoints;
    DataPoint* dataPoints = read_csv("../data/clustering/blobs_20d_10_include_y.csv", NUM_FEATURES, numPoints);
    if (!dataPoints) {
        std::cerr << "Failed to read CSV data" << std::endl;
        return 1;
    }

    if (!kmeans.loadFromDataPoints(dataPoints, numPoints)) {
        std::cerr << "Failed to load data points into KMeans" << std::endl;
        delete[] dataPoints;
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

    double* centroids = kmeans.getCentroids();
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            std::cout << centroids[i * NUM_FEATURES + j] << " ";
        }
        std::cout << std::endl;
    }

    delete[] dataPoints;

    return 0;
}