#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cmath>
#include <cblas.h>
#include <lapacke.h> // Include LAPACKE for dgesvd

using namespace std;
using namespace std::chrono;

// Function to read CSV file into a 1D vector (row-major)
vector<double> readCSV(const string& filename, int& rows, int& cols) {
    vector<double> data;
    
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        rows = 0;
        cols = 0;
        return data;
    }
    
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
        if (row.size() != cols) {
            cerr << "Error: Inconsistent number of columns in row " << rows + 1 << endl;
            rows = 0;
            cols = 0;
            return data;
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

class PCA {
public:
    vector<double> mean;
    int _n_components;
    vector<double> projected;
    vector<double> eigenvectors, eigenvalues;

    PCA(int n_components, int rows, int cols) {
        if (n_components <= 0) {
            cerr << "Error: Number of components must be positive" << endl;
            exit(1);
        }
        _n_components = min(n_components, cols);
        _n_components = min(_n_components, rows);
    }

    vector<double> transform(vector<double> data, int& rows, int& cols) {
        centerData(data, rows, cols, mean);
        
        computeEigenvectors(data, rows, cols, eigenvectors, eigenvalues, _n_components);
        
        projected.resize(rows * _n_components);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    rows, _n_components, cols, 1.0, data.data(), cols,
                    eigenvectors.data(), _n_components,
                    0.0, projected.data(), _n_components);
        
        vector<double> reconstructed(rows * cols);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    rows, cols, _n_components, 1.0,
                    projected.data(), _n_components,
                    eigenvectors.data(), _n_components,
                    0.0, reconstructed.data(), cols);

        // Validate projected data by computing variance of each component
        cout << "Variance of projected components:\n";
        for (int j = 0; j < _n_components; j++) {
            double mean = 0.0, variance = 0.0;
            for (int i = 0; i < rows; i++) {
                mean += projected[i * _n_components + j];
            }
            mean /= rows;
            for (int i = 0; i < rows; i++) {
                double diff = projected[i * _n_components + j] - mean;
                variance += diff * diff;
            }
            variance /= rows;
            cout << "Component " << j + 1 << ": " << variance << " (Eigenvalue: " << eigenvalues[j] << ")" << endl;
        }

        return reconstructed;
    }

    void centerData(vector<double>& data, int rows, int cols, vector<double>& mean) {
        mean.clear();
        cout << "Centering data...\n";
        for (int j = 0; j < cols; j++) {
            double mu = 0.0;
            for (int i = 0; i < rows; i++) {
                mu += data[i * cols + j];
            }
            mu /= rows;
            
            for (int i = 0; i < rows; i++) {
                data[i * cols + j] -= mu;
            }
            mean.push_back(mu);
        }

        // Validate centering
        cout << "Validating centering (mean of each column should be ~0):\n";
        for (int j = 0; j < cols; j++) {
            double post_center_mean = 0.0;
            for (int i = 0; i < rows; i++) {
                post_center_mean += data[i * cols + j];
            }
            post_center_mean /= rows;
            cout << "Column " << j + 1 << " mean after centering: " << post_center_mean << endl;
            if (abs(post_center_mean) > 1e-10) {
                cerr << "Warning: Column " << j + 1 << " is not centered properly (mean = " << post_center_mean << ")" << endl;
            }
        }
    }

private:
    void computeEigenvectors(const vector<double>& data, int rows, int cols, 
                            vector<double>& eigenvectors, vector<double>& eigenvalues, 
                            int n_components) {
        // Use SVD: data = U * S * V^T
        vector<double> data_copy = data; // Copy data since dgesvd modifies the input
        vector<double> U(rows * rows);   // Left singular vectors (not needed fully, but allocated)
        vector<double> S(cols);          // Singular values
        vector<double> VT(cols * cols);  // Right singular vectors (V^T)
        vector<double> superb(min(rows, cols) - 1); // Work array for dgesvd

        int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'A', rows, cols, data_copy.data(), cols,
                                  S.data(), U.data(), rows, VT.data(), cols, superb.data());
        if (info != 0) {
            cerr << "Error: LAPACKE_dgesvd failed with info = " << info << endl;
            exit(1);
        }

        // Eigenvalues of covariance matrix are (singular values)^2 / rows
        eigenvalues.resize(n_components);
        for (int i = 0; i < n_components; i++) {
            eigenvalues[i] = (S[i] * S[i]) / rows;
            cout << "Eigenvalue " << i + 1 << ": " << eigenvalues[i] << endl;
        }

        // Eigenvectors are the first n_components rows of V^T (columns of V)
        eigenvectors.resize(cols * n_components);
        for (int k = 0; k < n_components; k++) {
            for (int i = 0; i < cols; i++) {
                eigenvectors[k * cols + i] = VT[k * cols + i];
            }
        }

        // Validate orthogonality of eigenvectors
        for (int k = 0; k < n_components; k++) {
            for (int m = 0; m < k; m++) {
                double dot = 0.0;
                for (int i = 0; i < cols; i++) {
                    dot += eigenvectors[m * cols + i] * eigenvectors[k * cols + i];
                }
                cout << "Dot product of eigenvectors " << m + 1 << " and " << k + 1 << ": " << dot << endl;
            }
        }
    }
};

int main(int argc, char *argv[]) {
    int rows(0), cols(0);
    unsigned int n_components = 3;

    if (argc > 1) {
        n_components = atoi(argv[1]);
    }

    cout << "Reading data..." << endl;
    vector<double> data = readCSV("../data/blobs/X_20d_10.csv", rows, cols);
    
    if (rows <= 0 || cols <= 0) {
        cerr << "Error: Invalid data dimensions (rows = " << rows << ", cols = " << cols << ")" << endl;
        return 1;
    }
    
    PCA pca(n_components, rows, cols);
    
    auto start = high_resolution_clock::now();
    vector<double> reconstructed = pca.transform(data, rows, cols);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    double error = computeReconstructionError(data, reconstructed, rows, cols);
    
    writeCSV("../results/pca/projected_data.csv", pca.projected, rows, n_components);
    writeCSV("../results/pca/reconstructed_data.csv", reconstructed, rows, cols);
    
    ofstream result_file("../results/pca/results.csv");
    result_file << "execution_time_us,reconstruction_error_rmse\n";
    result_file << duration.count() << "," << error << "\n";
    result_file.close();
    
    cout << "Execution time: " << duration.count() << " microseconds\n";
    cout << "Reconstruction error (RMSE): " << error << endl;
    cout << "Number of components used: " << n_components << endl;
    
    return 0;
}