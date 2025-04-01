#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cmath>
#include <queue>

#include <algorithm>

using namespace std;
using namespace std::chrono;

vector<float> readCSV(const string& filename, int& rows, int& cols) {
    vector<float> data;
    
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        rows = 0;
        cols = 0;
        return data;
    }
    
    string line;
    rows = 0;
    bool first_row = true;
    
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<float> row;
        
        // Skip the first column (empty in header, index in data rows)
        getline(ss, value, ',');
        
        // Handle the header row
        if (first_row) {
            first_row = false;
            vector<string> headers;
            while (getline(ss, value, ',')) {
                headers.push_back(value);
            }
            cols = headers.size(); // Number of feature columns
            continue;
        }
        
        // Process data rows
        while (getline(ss, value, ',')) {
            if (value.empty()) {
                cerr << "Error: Empty value in row " << rows + 1 << endl;
                rows = 0;
                cols = 0;
                file.close();
                return data;
            }
            try {
                row.push_back(stod(value));
            } catch (const std::invalid_argument& e) {
                cerr << "Error: Invalid number '" << value << "' in row " << rows + 1 << endl;
                rows = 0;
                cols = 0;
                file.close();
                return data;
            }
        }
        
        if (row.size() != cols) {
            cerr << "Error: Inconsistent number of columns in row " << rows + 1 
                 << " (expected " << cols << ", got " << row.size() << ")" << endl;
            rows = 0;
            cols = 0;
            file.close();
            return data;
        }
        
        data.insert(data.end(), row.begin(), row.end());
        rows++;
    }
    
    file.close();
    return data;
}

void writeCSV(const string& filename, const vector<half_float::half>& data, int rows, int cols) {
    ofstream file(filename);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << data[i * cols + j];
            if (j < cols - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}


struct Neighbor { // Structure for priority queue (max-heap for largest distances)
    int index;
    float distance_sq;
    bool operator<(const Neighbor& other) const {
        return distance_sq < other.distance_sq; // Max-heap
    }
};

vector<pair<int, float>> kNearestNeighborSearch(const vector<float>& data, int n_samples, int n_features,
                                                 const vector<float>& query, int k, float& runtime) {
    auto start = high_resolution_clock::now();

    priority_queue<Neighbor> pq;
    for (int i = 0; i < n_samples; i++) {
        float distance_sq = 0.0;
        for (int j = 0; j < n_features; j++) {
            float diff = data[i * n_features + j] - query[j];
            distance_sq += diff * diff;
        }
        if (pq.size() < k) {
            pq.push({i, distance_sq});
        } else if (distance_sq < pq.top().distance_sq) {
            pq.pop();
            pq.push({i, distance_sq});
        }
    }

    vector<pair<int, float>> k_nearest(k);
    float max_val;
    float top_val;

    for (int i = k - 1; i >= 0; i--) {
        top_val = pq.top().distance_sq;
        max_val = max((float)0.0, top_val);
        k_nearest[i] = {pq.top().index, sqrt(max_val)};
        pq.pop();
    }

    auto end = high_resolution_clock::now();
    runtime = static_cast<half_float::half>(duration_cast<microseconds>(end - start).count()) / 1000.0;
    return k_nearest;
}

int main(int argc, char *argv[]) {
    int n_samples, n_features, n_queries, n_features_q, k_gt, n_features_gt;
    int k = 3; // Default number of neighbors

    cout << "Using half_float::half precision" << endl;

    // Read dataset
    cout << "Reading dataset..." << endl;
    vector<float> data = readCSV("dataset.csv", n_samples, n_features);
    if (n_samples <= 0 || n_features <= 0) {
        cerr << "Error: Invalid dataset dimensions" << endl;
        return 1;
    }
    if (k > n_samples) {
        k = n_samples;
        cout << "Warning: k adjusted to " << k << endl;
    }

    // Read queries
    cout << "Reading queries..." << endl;
    vector<float> queries = readCSV("queries.csv", n_queries, n_features_q);
    if (n_features_q != n_features) {
        cerr << "Error: Query feature dimension mismatch" << endl;
        return 1;
    }

    // Read ground truth
    cout << "Reading ground truth..." << endl;
    vector<float> gt_data = readCSV("ground_truth.csv", k_gt, n_features_gt);
    if (k_gt != n_queries || n_features_gt != k) {
        cerr << "Error: Ground truth dimensions mismatch" << endl;
        return 1;
    }
    vector<vector<int>> ground_truth(n_queries, vector<int>(k));
    for (int i = 0; i < n_queries; i++) {
        for (int j = 0; j < k; j++) {
            ground_truth[i][j] = static_cast<int>(gt_data[i * k + j]);
        }
    }

    // Process queries
    vector<float> all_results;
    vector<half_float::half> runtimes(n_queries), accuracies(n_queries);
    for (int q = 0; q < n_queries; q++) {
        vector<float> query(n_features);
        for (int j = 0; j < n_features; j++) {
            query[j] = queries[q * n_features + j];
        }

        float runtime;
        auto k_nearest = kNearestNeighborSearch(data, n_samples, n_features, query, k, runtime);
        runtimes[q] = runtime;

        cout << "Query " << q + 1 << " (half_float::half):\n";
        for (int i = 0; i < k; i++) {
            cout << "Neighbor " << i + 1 << ": Index = " << k_nearest[i].first 
                 << ", Distance = " << k_nearest[i].second << endl;
            all_results.push_back(static_cast<half_float::half>(k_nearest[i].first));
            all_results.push_back(k_nearest[i].second);
        }

        int correct = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                if (k_nearest[i].first == ground_truth[q][j]) {
                    correct++;
                    break;
                }
            }
        }
        accuracies[q] = static_cast<half_float::half>(correct) / k;

        cout << "Runtime: " << runtime << " ms, Accuracy: " << accuracies[q] << endl;
    }

    return 0;
}