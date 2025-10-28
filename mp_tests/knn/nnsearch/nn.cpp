#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <queue>
#include <cmath>
#include <cstring>

using namespace std;
using namespace std::chrono;

#define MAX_SAMPLES 10000
#define MAX_FEATURES 100
#define MAX_QUERIES 1000
#define MAX_K 10
#define MAX_LINE 4096 

struct Neighbor { 
    int index;
    __PROMISE__ distance_sq;
    bool operator<(const Neighbor& other) const {
        return distance_sq < other.distance_sq; // Max-heap
    }
};

struct NeighborResult { 
    int index;
    __PROMISE__ distance;
};

NeighborResult* kNearestNeighborSearch(const __PROMISE__* data, int n_samples, int n_features,
                                      const __PROMISE__* query, int k) {

    priority_queue<Neighbor> pq;
    for (int i = 0; i < n_samples; i++) {
        __PROMISE__ distance_sq = 0.0;
        for (int j = 0; j < n_features; j++) {
            __PROMISE__ diff = data[i * n_features + j] - query[j];
            distance_sq += diff * diff;
        }
        if (pq.size() < k) {
            pq.push({i, distance_sq});
        } else if (distance_sq < pq.top().distance_sq) {
            pq.pop();
            pq.push({i, distance_sq});
        }
    }

    NeighborResult* k_nearest = new NeighborResult[k];
    for (int i = k - 1; i >= 0; i--) {
        k_nearest[i].index = pq.top().index;
        double temp = 0;
        k_nearest[i].distance = sqrt(max(temp, pq.top().distance_sq));
        pq.pop();
    }

    return k_nearest;
}

__PROMISE__* readCSV(const string& filename, int& rows, int& cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        rows = 0;
        cols = 0;
        return nullptr;
    }

    char line[MAX_LINE];
    rows = 0;
    bool first_row = true;
    __PROMISE__* data = nullptr;
    int data_size = 0;

    while (file.getline(line, MAX_LINE)) {
        char* token = strtok(line, ","); // Skip first column
        if (!token) continue;

        if (first_row) {
            first_row = false;
            cols = 0;
            while ((token = strtok(nullptr, ","))) cols++;
            continue;
        }

        __PROMISE__ row_data[MAX_FEATURES];
        int col_count = 0;
        while ((token = strtok(nullptr, ","))) {
            if (col_count >= MAX_FEATURES) {
                cerr << "Error: Too many features in row " << rows + 1 << endl;
                rows = 0;
                cols = 0;
                file.close();
                delete[] data;
                return nullptr;
            }
            try {
                row_data[col_count++] = stod(token);
            } catch (const std::invalid_argument& e) {
                cerr << "Error: Invalid number '" << token << "' in row " << rows + 1 << endl;
                rows = 0;
                cols = 0;
                file.close();
                delete[] data;
                return nullptr;
            }
        }

        if (rows == 0) {
            cols = col_count; // Set cols based on first data row
            data = new __PROMISE__[MAX_SAMPLES * cols];
        } else if (col_count != cols) {
            cerr << "Error: Inconsistent number of columns in row " << rows + 1
                 << " (expected " << cols << ", got " << col_count << ")" << endl;
            rows = 0;
            cols = 0;
            file.close();
            delete[] data;
            return nullptr;
        }

        for (int j = 0; j < cols; j++) {
            data[rows * cols + j] = row_data[j];
        }
        rows++;
        if (rows >= MAX_SAMPLES) {
            cerr << "Error: Too many samples" << endl;
            rows = 0;
            cols = 0;
            file.close();
            delete[] data;
            return nullptr;
        }
    }

    file.close();
    return data;
}

int main(int argc, char *argv[]) {
    int n_samples, n_features, n_queries, n_features_q, k_gt, n_features_gt;
    int k = 3; 
    // Read dataset
    cout << "Reading dataset..." << endl;
    __PROMISE__* data = readCSV("dataset.csv", n_samples, n_features);
    if (n_samples <= 0 || n_features <= 0) {
        cerr << "Error: Invalid dataset dimensions" << endl;
        delete[] data;
        return 1;
    }
    if (k > n_samples) {
        k = n_samples;
        cout << "Warning: k adjusted to " << k << endl;
    }

    // Read queries
    cout << "Reading queries..." << endl;
    __PROMISE__* queries = readCSV("queries.csv", n_queries, n_features_q);
    if (n_features_q != n_features) {
        cerr << "Error: Query feature dimension mismatch" << endl;
        delete[] data;
        delete[] queries;
        return 1;
    }

    // Read ground truth
    cout << "Reading ground truth..." << endl;
    __PROMISE__* gt_data = readCSV("ground_truth.csv", k_gt, n_features_gt);
    if (k_gt != n_queries || n_features_gt != k) {
        cerr << "Error: Ground truth dimensions mismatch" << endl;
        delete[] data;
        delete[] queries;
        delete[] gt_data;
        return 1;
    }
    int ground_truth[MAX_QUERIES][MAX_K];
    for (int i = 0; i < n_queries; i++) {
        for (int j = 0; j < k; j++) {
            ground_truth[i][j] = static_cast<int>(gt_data[i * k + j]);
        }
    }
    delete[] gt_data;

    // Process queries
    __PROMISE__* all_results = new __PROMISE__[n_queries * k * 2];
    __PROMISE__  accuracies[MAX_QUERIES];
    for (int q = 0; q < n_queries; q++) {
        __PROMISE__ query[MAX_FEATURES];
        for (int j = 0; j < n_features; j++) {
            query[j] = queries[q * n_features + j];
        }

        NeighborResult* k_nearest = kNearestNeighborSearch(data, n_samples, n_features, query, k);

        for (int i = 0; i < k; i++) {
            cout << "Neighbor " << i + 1 << ": Index = " << k_nearest[i].index
                 << ", Distance = " << k_nearest[i].distance << endl;
            all_results[q * k * 2 + i * 2] = static_cast<double>(k_nearest[i].index);
            all_results[q * k * 2 + i * 2 + 1] = k_nearest[i].distance;
        }

        int correct = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                if (k_nearest[i].index == ground_truth[q][j]) {
                    correct++;
                    break;
                }
            }
        }
        accuracies[q] = static_cast<double>(correct) / k;

        delete[] k_nearest;
    }

    PROMISE_CHECK_ARRAY(all_results, n_queries * k * 2);
    // PROMISE_CHECK_ARRAY(accuracies, MAX_QUERIES);
    delete[] data;
    delete[] queries;
    delete[] all_results;

    return 0;
}