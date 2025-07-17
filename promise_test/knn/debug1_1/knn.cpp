#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>
#include <ctime>



struct DataPoint {
    float* features;
    int feature_count;
    int label;
    
    DataPoint() : features(nullptr), feature_count(0), label(0) {}
    
    ~DataPoint() {
        delete[] features;
    }
};

DataPoint* read_csv(const std::string& filename, int& data_size) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        data_size = 0;
        return nullptr;
    }

    std::string line;
    getline(file, line);
    
    data_size = 0;
    while (getline(file, line)) data_size++;
    file.clear();
    file.seekg(0);
    getline(file, line);
    
    DataPoint* data = new DataPoint[data_size];
    int current_index = 0;
    
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        
        getline(ss, value, ',');
        
        int feature_count = 0;
        std::stringstream feature_ss(line);
        getline(feature_ss, value, ',');
        while (getline(feature_ss, value, ',')) feature_count++;
        feature_count--;
        
        float* features = new float[feature_count];
        ss.clear();
        ss.str(line);
        getline(ss, value, ',');
        
        int i = 0;
        while (getline(ss, value, ',')) {
            if (i < feature_count) {
                features[i] = std::stod(value);
            } else {
                data[current_index].label = std::stoi(value);
            }
            i++;
        }
        
        data[current_index].features = features;
        data[current_index].feature_count = feature_count;
        current_index++;
    }
    
    std::cout << "Loaded " << data_size << " data points with " 
              << (data_size > 0 ? data[0].feature_count : 0) << " features each" << std::endl;
    
    file.close();
    return data;
}



// Calculate Euclidean distance between two data points
float euclidean_distance(const DataPoint& a, const DataPoint& b) {
    float sum = 0.0;
    for (int i = 0; i < a.feature_count; ++i) {
        sum += (a.features[i] - b.features[i]) * (a.features[i] - b.features[i]);
    }
    return sqrt(sum);
}


struct Neighbor { // Structure to store distance and index
    flx::floatx<4, 3> distance;
    int index;
};

// Comparison function for sorting neighbors
bool compare_neighbors(const Neighbor& a, const Neighbor& b) {
    return a.distance < b.distance;
}

// kNN prediction for a single test point
int knn_predict(const DataPoint* train_data, int train_size, const DataPoint& test_point, int k) {
    Neighbor* neighbors = new Neighbor[train_size];
    
    // Calculate distances
    for (int i = 0; i < train_size; ++i) {
        neighbors[i].distance = euclidean_distance(train_data[i], test_point);
        neighbors[i].index = i;
    }
    
    // Sort neighbors by distance
    std::sort(neighbors, neighbors + train_size, compare_neighbors);
    
    // Count votes for each class
    int* class_votes = new int[3](); // Iris has 3 classes
    for (int i = 0; i < k; ++i) {
        int label = train_data[neighbors[i].index].label;
        class_votes[label]++;
    }
    
    // Find class with maximum votes
    int max_votes = 0;
    int predicted_class = 0;
    for (int i = 0; i < 3; ++i) {
        if (class_votes[i] > max_votes) {
            max_votes = class_votes[i];
            predicted_class = i;
        }
    }
    
    delete[] neighbors;
    delete[] class_votes;
    return predicted_class;
}


void train_test_split(DataPoint* data, int data_size, DataPoint*& train_data, int& train_size,
                      DataPoint*& test_data, int& test_size, float train_ratio) {
    train_size = static_cast<int>(data_size * train_ratio);
    test_size = data_size - train_size;
    
    // Shuffle indices
    int* indices = new int[data_size];
    for (int i = 0; i < data_size; ++i) indices[i] = i;
    
    std::mt19937 g(42);
    for (int i = data_size - 1; i > 0; --i) {
        std::uniform_int_distribution<> dis(0, i);
        int j = dis(g);
        std::swap(indices[i], indices[j]);
    }
    
    train_data = new DataPoint[train_size];
    test_data = new DataPoint[test_size];
    
    for (int i = 0; i < train_size; ++i) {
        train_data[i].feature_count = data[indices[i]].feature_count;
        train_data[i].label = data[indices[i]].label;
        train_data[i].features = new float[train_data[i].feature_count];
        for (int j = 0; j < train_data[i].feature_count; ++j) {
            train_data[i].features[j] = data[indices[i]].features[j];
        }
    }
    
    for (int i = 0; i < test_size; ++i) {
        test_data[i].feature_count = data[indices[train_size + i]].feature_count;
        test_data[i].label = data[indices[train_size + i]].label;
        test_data[i].features = new float[test_data[i].feature_count];
        for (int j = 0; j < test_data[i].feature_count; ++j) {
            test_data[i].features[j] = data[indices[train_size + i]].features[j];
        }
    }
    
    delete[] indices;
}

int main() {
    int data_size;
    DataPoint* data = read_csv("iris.csv", data_size);
    if (!data) return 1;
    
    // Perform train-test split (80-20)
    DataPoint* train_data;
    DataPoint* test_data;
    int train_size, test_size;
    train_test_split(data, data_size, train_data, train_size, test_data, test_size, 0.8);
    
    int k = 3;
    int* predictions = new int[test_size];
    for (int i = 0; i < test_size; ++i) {
        predictions[i] = knn_predict(train_data, train_size, test_data[i], k);
    }
    
    int correct = 0;
    for (int i = 0; i < test_size; ++i) {
        if (predictions[i] == test_data[i].label) {
            correct++;
        }
    }
    float accuracy = static_cast<flx::floatx<4, 3>>(correct) / test_size * 100;
    
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    
    delete[] predictions;
    delete[] train_data;
    delete[] test_data;
    delete[] data;
    
    return 0;
}