#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <unordered_map>

struct DataPoint {
    __PROMISE__* features; 
    int label;
    size_t numFeatures;

    DataPoint() : features(nullptr), label(0), numFeatures(0) {}

    DataPoint(size_t numFeatures_) : numFeatures(numFeatures_), label(0) {
        features = new __PROMISE__[numFeatures];
    }

    ~DataPoint() {
        delete[] features;
    }

    DataPoint(const DataPoint& other) : numFeatures(other.numFeatures), label(other.label) {
        features = new __PROMISE__[numFeatures];
        for (size_t i = 0; i < numFeatures; ++i) {
            features[i] = other.features[i];
        }
    }

    DataPoint& operator=(const DataPoint& other) {
        if (this != &other) {
            delete[] features;
            numFeatures = other.numFeatures;
            label = other.label;
            features = new __PROMISE__[numFeatures];
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
        data[idx].features = new __PROMISE__[numFeatures];
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
    __PROMISE__* data;          // numPoints * numFeatures
    int* labels;           // numPoints
    int* groundTruth;      // numPoints
    __PROMISE__* centroids;     // k * numFeatures
    __PROMISE__ runtime;

    void getPoint(int idx, __PROMISE__* point) const {
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

    __PROMISE__ euclideanDistance(const __PROMISE__* p1, const __PROMISE__* p2) const {
        __PROMISE__ sum = 0.0;
        for (int i = 0; i < numFeatures; ++i) {
            sum += (p1[i] - p2[i]) * (p1[i] - p2[i]);
        }
        return sqrt(sum);
    }

    void initializeCentroids() {
        // Deterministically select the first k points as initial centroids
        for (int c = 0; c < k; ++c) {
            if (c >= numPoints) {
                std::cerr << "Not enough points to select " << k << " centroids" << std::endl;
                return;
            }
            __PROMISE__* tempPoint = new __PROMISE__[numFeatures];
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

        data = new __PROMISE__[numPoints * numFeatures];
        groundTruth = new int[numPoints];
        labels = new int[numPoints];
        centroids = new __PROMISE__[k * numFeatures];

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

        __PROMISE__* point = new __PROMISE__[numFeatures];
        __PROMISE__* centroid = new __PROMISE__[numFeatures];

        while (changed && iterations < maxIterations) {
            changed = false;

            for (int i = 0; i < numPoints; ++i) {
                getPoint(i, point);
                __PROMISE__ minDist = 9999;
                int newLabel = 0;

                for (int c = 0; c < k; ++c) {
                    for (int j = 0; j < numFeatures; ++j) {
                        centroid[j] = centroids[c * numFeatures + j];
                    }
                    __PROMISE__ dist = euclideanDistance(point, centroid);
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
        runtime = std::chrono::duration<__PROMISE__>(end - start).count();

        std::cout << "Converged after " << iterations << " iterations" << std::endl;
        std::cout << "Runtime: " << runtime << " seconds" << std::endl;
    }



    const int* getLabels() const { return labels; }
    __PROMISE__* getCentroids() const { return centroids; }
    __PROMISE__ getRuntime() const { return runtime; }
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

    size_t numPoints;
    DataPoint* dataPoints = read_csv("blobs_20d_10_include_y_20000.csv", NUM_FEATURES, numPoints);
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

    __PROMISE__* centroids = kmeans.getCentroids();
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < NUM_FEATURES; ++j) {
            std::cout << centroids[i * NUM_FEATURES + j] << " ";
        }
        std::cout << std::endl;
    }

    PROMISE_CHECK_ARRAY(centroids,  NUM_FEATURES*K);
    delete[] dataPoints;

    return 0;
}