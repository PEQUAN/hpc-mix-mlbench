#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>

using namespace std;
using namespace std::chrono;

// Function to read CSV file into a 1D vector (row-major)
vector<double> readCSV(const string& filename, int& rows, int& cols) {
    vector<double> data;
    ifstream file(filename);
    string line;
    
    rows = 0;
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> row;
        
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }
        
        if (rows == 0) {
            cols = row.size();
        }
        
        data.insert(data.end(), row.begin(), row.end());
        rows++;
    }
    
    file.close();
    return data;
}

// Function to write matrix to CSV
void writeCSV(const string& filename, const vector<double>& data, int rows, int cols) {
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

// Function to center the data
void centerData(vector<double>& data, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        double mean = 0.0;
        for (int i = 0; i < rows; i++) {
            mean += data[i * cols + j];
        }
        mean /= rows;
        
        for (int i = 0; i < rows; i++) {
            data[i * cols + j] -= mean;
        }
    }
}

// Compute reconstruction error
double computeReconstructionError(const vector<double>& original, 
                                const vector<double>& reconstructed, 
                                int rows, int cols) {
    double error = 0.0;
    for (int i = 0; i < rows * cols; i++) {
        double diff = original[i] - reconstructed[i];
        error += diff * diff;
    }
    return sqrt(error / (rows * cols));
}

int main() {
    int rows, cols;
    
    // Read data
    cout << "Reading data..." << endl;
    vector<double> data = readCSV("../data/blobs/X_20d_10.csv", rows, cols);
    
    // Center the data
    centerData(data, rows, cols);
    
    // Start timing
    auto start = high_resolution_clock::now();
    
    // Compute covariance matrix
    vector<double> cov(cols * cols);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                cols, cols, rows, 1.0/rows,
                data.data(), cols,
                data.data(), cols,
                0.0, cov.data(), cols);
    
    // Compute eigenvalues and eigenvectors using LAPACKE
    vector<double> eigenvalues(cols);
    vector<double> eigenvectors(cols * cols);
    
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', cols,
                 cov.data(), cols, eigenvalues.data());
    
    // Copy eigenvectors (LAPACKE stores them in cov)
    copy(cov.begin(), cov.end(), eigenvectors.begin());
    
    // Select number of components (let's keep 90% of variance)
    double total_variance = 0.0;
    for (double val : eigenvalues) total_variance += val;
    
    double cumulative_variance = 0.0;
    int n_components = 0;
    for (int i = cols-1; i >= 0; i--) {
        cumulative_variance += eigenvalues[i];
        n_components++;
        if (cumulative_variance / total_variance >= 0.9) break;
    }
    
    // Project data onto principal components
    vector<double> projected(rows * n_components);
    vector<double> selected_eigenvectors(n_components * cols);
    
    // Select top n_components eigenvectors
    for (int i = 0; i < n_components; i++) {
        for (int j = 0; j < cols; j++) {
            selected_eigenvectors[i * cols + j] = eigenvectors[(cols-1-i) * cols + j];
        }
    }
    
    // Project data
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, n_components, cols, 1.0,
                data.data(), cols,
                selected_eigenvectors.data(), cols,
                0.0, projected.data(), n_components);
    
    // Reconstruct data
    vector<double> reconstructed(rows * cols);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rows, cols, n_components, 1.0,
                projected.data(), n_components,
                selected_eigenvectors.data(), cols,
                0.0, reconstructed.data(), cols);
    
    // End timing
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    // Compute reconstruction error
    double error = computeReconstructionError(data, reconstructed, rows, cols);
    
    // Save results
    writeCSV("projected_data.csv", projected, rows, n_components);
    writeCSV("reconstructed_data.csv", reconstructed, rows, cols);
    
    // Save timing and error
    ofstream result_file("results.csv");
    result_file << "execution_time_us,reconstruction_error_rmse\n";
    result_file << duration.count() << "," << error << "\n";
    result_file.close();
    
    cout << "Execution time: " << duration.count() << " microseconds\n";
    cout << "Reconstruction error (RMSE): " << error << endl;
    cout << "Number of components used: " << n_components << endl;
    
    return 0;
}