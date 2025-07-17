#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

struct Matrix {
    int rows, cols;
    double* data;

    Matrix() : rows(0), cols(0), data(nullptr) {}

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(nullptr) {
        if (other.data && rows > 0 && cols > 0) {
            data = new double[rows * cols];
            std::memcpy(data, other.data, rows * cols * sizeof(double));
        }
    }

    Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data) {
        other.rows = 0;
        other.cols = 0;
        other.data = nullptr;
    }

    void assign(const Matrix& other) {
        if (this != &other) {
            if (data) {
                delete[] data;
            }
            rows = other.rows;
            cols = other.cols;
            data = nullptr;
            if (other.data && rows > 0 && cols > 0) {
                data = new double[rows * cols];
                std::memcpy(data, other.data, rows * cols * sizeof(double));
            }
        }
    }

    void assign(Matrix&& other) noexcept {
        if (this != &other) {
            if (data) {
                delete[] data;
            }
            rows = other.rows;
            cols = other.cols;
            data = other.data;
            other.rows = 0;
            other.cols = 0;
            other.data = nullptr;
        }
    }

    ~Matrix() {
        if (data) {
            delete[] data;
            data = nullptr;
        }
        rows = 0;
        cols = 0;
    }
};

Matrix create_matrix(int r, int c) {
    if (r < 0 || c < 0) {
        std::cerr << "Error: Invalid matrix dimensions (" << r << "x" << c << ")" << std::endl;
        return Matrix();
    }
    Matrix A;
    A.rows = r;
    A.cols = c;
    if (r > 0 && c > 0) {
        try {
            A.data = new double[r * c]();
        } catch (const std::bad_alloc&) {
            std::cerr << "Error: Failed to allocate matrix (" << r << "x" << c << ")" << std::endl;
            A.rows = 0;
            A.cols = 0;
            A.data = nullptr;
        }
    }
    return A;
}

void free_matrix(Matrix& A) {
    if (A.data) {
        delete[] A.data;
        A.data = nullptr;
    }
    A.rows = 0;
    A.cols = 0;
}

Matrix matrix_multiply(const Matrix& A, const Matrix& B) {
    if (!A.data || !B.data || A.cols != B.rows) {
        std::cerr << "Matrix multiplication error: incompatible dimensions ("
                  << A.rows << "x" << A.cols << ") * (" << B.rows << "x" << B.cols << ")" << std::endl;
        return create_matrix(0, 0);
    }
    Matrix result = create_matrix(A.rows, B.cols);
    if (!result.data) return result;
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < A.cols; ++k) {
                sum += A.data[i * A.cols + k] * B.data[k * B.cols + j];
            }
            result.data[i * result.cols + j] = sum;
        }
    }
    return result;
}

Matrix transpose(const Matrix& A) {
    if (!A.data) {
        std::cerr << "Error: Cannot transpose empty matrix" << std::endl;
        return create_matrix(0, 0);
    }
    Matrix result = create_matrix(A.cols, A.rows);
    if (!result.data) return result;
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            result.data[j * result.cols + i] = A.data[i * A.cols + j];
        }
    }
    return result;
}

Matrix generate_random_matrix(int n_samples, int n_features, double sparsity = 0.8, unsigned int seed = 42) {
    if (sparsity < 0.0 || sparsity > 1.0) {
        std::cerr << "Error: Sparsity must be between 0.0 and 1.0" << std::endl;
        return create_matrix(0, 0);
    }
    if (n_samples <= 0 || n_features <= 0) {
        std::cerr << "Error: Invalid dimensions n_samples=" << n_samples << ", n_features=" << n_features << std::endl;
        return create_matrix(0, 0);
    }

    Matrix data = create_matrix(n_samples, n_features);
    if (!data.data) {
        std::cerr << "Error: Failed to allocate matrix for data" << std::endl;
        return data;
    }

    std::mt19937 gen(seed);
    std::uniform_real_distribution<> value_dis(-10.0, 10.0);
    std::uniform_real_distribution<> sparsity_dis(0.0, 1.0);

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            if (sparsity_dis(gen) >= sparsity) {
                data.data[i * data.cols + j] = value_dis(gen);
            }
        }
    }

    return data;
}

Matrix scale_matrix(const Matrix& input) {
    if (!input.data || input.rows <= 0 || input.cols <= 0) {
        std::cerr << "Error: Invalid input matrix for scaling" << std::endl;
        return create_matrix(0, 0);
    }

    Matrix scaled = create_matrix(input.rows, input.cols);
    if (!scaled.data) return scaled;

    double* means = new double[input.cols]();
    double* stds = new double[input.cols]();
    int* counts = new int[input.cols]();

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            double val = input.data[i * input.cols + j];
            scaled.data[i * scaled.cols + j] = val;
            if (val != 0.0) {
                means[j] += val;
                counts[j]++;
            }
        }
    }
    for (int j = 0; j < input.cols; ++j) {
        means[j] = counts[j] > 0 ? means[j] / counts[j] : 0.0;
    }

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            double val = input.data[i * input.cols + j];
            if (val != 0.0) {
                double diff = val - means[j];
                stds[j] += diff * diff;
            }
        }
    }
    for (int j = 0; j < input.cols; ++j) {
        stds[j] = counts[j] > 0 ? std::sqrt(stds[j] / counts[j]) : 1.0;
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            double val = input.data[i * input.cols + j];
            if (val != 0.0) {
                scaled.data[i * scaled.cols + j] = (val - means[j]) / stds[j];
            }
        }
    }

    delete[] means;
    delete[] stds;
    delete[] counts;
    return scaled;
}

void writeCSV(const std::string& filename, const Matrix& data) {
    if (!data.data) {
        std::cerr << "Error: Cannot write empty matrix to " << filename << std::endl;
        return;
    }
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing" << std::endl;
        return;
    }
    for (int i = 0; i < data.rows; ++i) {
        for (int j = 0; j < data.cols; ++j) {
            file << data.data[i * data.cols + j];
            if (j < data.cols - 1) file << ",";
        }
        file << "\n";
    }
    file.close();
}

void computeReconstructionError(const Matrix& original, const Matrix& reconstructed, double& rmse, double& frobenius) {
    if (!original.data || !reconstructed.data || original.rows != reconstructed.rows || original.cols != reconstructed.cols) {
        std::cerr << "Dimension mismatch or invalid data in reconstruction error calculation: original ("
                  << original.rows << "x" << original.cols << "), reconstructed ("
                  << reconstructed.rows << "x" << reconstructed.cols << ")" << std::endl;
        rmse = frobenius = -1.0;
        return;
    }
    double error = 0.0;
    for (int i = 0; i < original.rows; ++i) {
        for (int j = 0; j < original.cols; ++j) {
            double diff = original.data[i * original.cols + j] - reconstructed.data[i * reconstructed.cols + j];
            error += diff * diff;
        }
    }
    frobenius = std::sqrt(error);
    rmse = std::sqrt(error / (original.rows * original.cols));
}

// QR Decomposition using Householder Reflections
void qr_decomposition(const Matrix& A, Matrix& Q, Matrix& R) {
    if (!A.data || A.rows != A.cols) {
        std::cerr << "Error: Invalid matrix for QR decomposition ("
                  << A.rows << "x" << A.cols << ")" << std::endl;
        free_matrix(Q);
        free_matrix(R);
        return;
    }
    int n = A.rows;
    Q.assign(create_matrix(n, n));
    R.assign(create_matrix(n, n));
    if (!Q.data || !R.data) {
        std::cerr << "Error: Failed to allocate Q or R matrices" << std::endl;
        free_matrix(Q);
        free_matrix(R);
        return;
    }

    // Initialize R as a copy of A
    for (int i = 0; i < n * n; ++i) {
        R.data[i] = A.data[i];
    }
    // Initialize Q as identity
    for (int i = 0; i < n; ++i) {
        Q.data[i * Q.cols + i] = 1.0;
    }

    Matrix v = create_matrix(n, 1); // Householder vector
    if (!v.data) {
        free_matrix(v);
        free_matrix(Q);
        free_matrix(R);
        return;
    }

    for (int j = 0; j < n; ++j) {
        // Extract subcolumn from j to n-1 in column j
        int len = n - j;
        for (int i = 0; i < len; ++i) {
            v.data[i] = R.data[(i + j) * R.cols + j];
        }
        for (int i = len; i < n; ++i) {
            v.data[i] = 0.0;
        }

        // Compute norm of the subcolumn
        double norm = 0.0;
        for (int i = 0; i < len; ++i) {
            norm += v.data[i] * v.data[i];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-10) {
            std::cerr << "Warning: Zero norm in Householder QR at column " << j
                      << ", indicating rank deficiency. Continuing with zero reflection." << std::endl;
            continue; // Skip reflection for zero column
        }

        // Compute Householder vector: v = x - ||x|| e_1
        double alpha = (v.data[0] >= 0 ? -norm : norm); // Choose sign to avoid cancellation
        v.data[0] -= alpha;
        double v_norm = 0.0;
        for (int i = 0; i < len; ++i) {
            v_norm += v.data[i] * v.data[i];
        }
        v_norm = std::sqrt(v_norm);
        if (v_norm < 1e-10) {
            continue; // Skip if v is zero after adjustment
        }
        for (int i = 0; i < n; ++i) {
            v.data[i] /= v_norm;
        }

        // Apply Householder reflection to R: R = (I - 2vv^T)R
        for (int k = j; k < n; ++k) { // Update columns j to n-1
            double dot = 0.0;
            for (int i = j; i < n; ++i) {
                dot += v.data[i - j] * R.data[i * R.cols + k];
            }
            for (int i = j; i < n; ++i) {
                R.data[i * R.cols + k] -= 2.0 * dot * v.data[i - j];
            }
        }

        // Update Q: Q = Q (I - 2vv^T)
        for (int k = 0; k < n; ++k) { // Update all columns of Q
            double dot = 0.0;
            for (int i = j; i < n; ++i) {
                dot += v.data[i - j] * Q.data[i * Q.cols + k];
            }
            for (int i = j; i < n; ++i) {
                Q.data[i * Q.cols + k] -= 2.0 * dot * v.data[i - j];
            }
        }
    }

    free_matrix(v);
}

class PCA {
private:
    int n_components;
    Matrix mean;
    Matrix projected;
    Matrix eigenvectors;
    Matrix eigenvalues;

    void computeEigenvectors(Matrix& data, int rows, int cols) {
        if (!data.data) {
            std::cerr << "Error: Invalid data matrix for eigenvector computation" << std::endl;
            return;
        }
        Matrix Xt = transpose(data);
        Matrix cov = matrix_multiply(Xt, data);
        free_matrix(Xt);
        if (!cov.data) {
            std::cerr << "Error: Failed to compute covariance matrix" << std::endl;
            return;
        }
        for (int i = 0; i < cov.rows; ++i) {
            for (int j = 0; j < cov.cols; ++j) {
                cov.data[i * cov.cols + j] /= rows;
            }
        }

        Matrix A = create_matrix(cols, cols);
        A.assign(cov);
        Matrix Q_acc = create_matrix(cols, cols);
        if (!Q_acc.data || !A.data) {
            std::cerr << "Error: Failed to allocate matrices for QR algorithm" << std::endl;
            free_matrix(A);
            free_matrix(Q_acc);
            free_matrix(cov);
            return;
        }
        for (int i = 0; i < cols; ++i) {
            Q_acc.data[i * Q_acc.cols + i] = 1.0;
        }

        int max_iter = 100;
        for (int iter = 0; iter < max_iter; ++iter) {
            Matrix Q, R;
            qr_decomposition(A, Q, R);
            if (!Q.data || !R.data) {
                std::cerr << "Error: QR decomposition failed at iteration " << iter << std::endl;
                free_matrix(A);
                free_matrix(Q_acc);
                free_matrix(cov);
                return;
            }
            Matrix RQ = matrix_multiply(R, Q);
            if (!RQ.data) {
                std::cerr << "Error: Matrix multiplication failed in QR algorithm" << std::endl;
                free_matrix(Q);
                free_matrix(R);
                free_matrix(A);
                free_matrix(Q_acc);
                free_matrix(cov);
                return;
            }
            A.assign(RQ);
            Matrix Q_acc_new = matrix_multiply(Q_acc, Q);
            if (!Q_acc_new.data) {
                std::cerr << "Error: Accumulator update failed in QR algorithm" << std::endl;
                free_matrix(Q);
                free_matrix(R);
                free_matrix(RQ);
                free_matrix(A);
                free_matrix(Q_acc);
                free_matrix(cov);
                return;
            }
            Q_acc.assign(Q_acc_new);
            free_matrix(Q);
            free_matrix(R);
            free_matrix(RQ);
            free_matrix(Q_acc_new);
        }

        eigenvalues.assign(create_matrix(1, n_components));
        eigenvectors.assign(create_matrix(cols, n_components));
        if (!eigenvalues.data || !eigenvectors.data) {
            std::cerr << "Error: Failed to allocate eigenvalues or eigenvectors" << std::endl;
            free_matrix(A);
            free_matrix(Q_acc);
            free_matrix(cov);
            return;
        }

        struct EigenPair {
            double value;
            int index;
        };
        std::vector<EigenPair> eigen_pairs(cols);
        int valid_pairs = 0;
        for (int i = 0; i < cols; ++i) {
            if (std::abs(A.data[i * A.cols + i]) > 1e-10) {
                eigen_pairs[valid_pairs] = {A.data[i * A.cols + i], i};
                valid_pairs++;
            }
        }
        if (valid_pairs < n_components) {
            std::cerr << "Warning: Only " << valid_pairs << " non-zero eigenvalues found, but "
                      << n_components << " components requested. Adjusting n_components." << std::endl;
            n_components = valid_pairs;
            eigenvalues.assign(create_matrix(1, n_components));
            eigenvectors.assign(create_matrix(cols, n_components));
            if (!eigenvalues.data || !eigenvectors.data) {
                std::cerr << "Error: Failed to reallocate eigenvalues or eigenvectors" << std::endl;
                free_matrix(A);
                free_matrix(Q_acc);
                free_matrix(cov);
                return;
            }
        }
        eigen_pairs.resize(valid_pairs);
        std::sort(eigen_pairs.begin(), eigen_pairs.end(),
                  [](const EigenPair& a, const EigenPair& b) { return std::abs(a.value) > std::abs(b.value); });

        for (int k = 0; k < n_components; ++k) {
            int idx = eigen_pairs[k].index;
            eigenvalues.data[k] = eigen_pairs[k].value;
            for (int i = 0; i < cols; ++i) {
                eigenvectors.data[i * eigenvectors.cols + k] = Q_acc.data[i * Q_acc.cols + idx];
            }
            std::cout << "Eigenvalue " << k + 1 << ": " << eigenvalues.data[k] << std::endl;
        }

        free_matrix(A);
        free_matrix(Q_acc);
        free_matrix(cov);

        for (int k = 0; k < n_components; ++k) {
            for (int m = 0; m < k; ++m) {
                double dot = 0.0;
                for (int i = 0; i < cols; ++i) {
                    dot += eigenvectors.data[i * eigenvectors.cols + k] * eigenvectors.data[i * eigenvectors.cols + m];
                }
                std::cout << "Dot product of eigenvectors " << m + 1 << " and " << k + 1 << ": " << dot << std::endl;
            }
        }
    }

public:
    PCA(int n_components, int rows, int cols) : n_components(n_components),
        mean(create_matrix(1, cols)),
        projected(create_matrix(0, 0)),
        eigenvectors(create_matrix(0, 0)),
        eigenvalues(create_matrix(0, 0)) {
        if (n_components <= 0 || n_components > cols || n_components > rows) {
            std::cerr << "Error: Invalid number of components: " << n_components
                      << " (must be positive and <= min(rows, cols) = " << std::min(rows, cols) << ")" << std::endl;
            exit(1);
        }
    }

    ~PCA() {
        free_matrix(mean);
        free_matrix(projected);
        free_matrix(eigenvectors);
        free_matrix(eigenvalues);
    }

    Matrix transform(Matrix& data) {
        if (!data.data) {
            std::cerr << "Error: Invalid input data for transform" << std::endl;
            return create_matrix(0, 0);
        }
        centerData(data);

        computeEigenvectors(data, data.rows, data.cols);
        if (!eigenvectors.data) {
            std::cerr << "Error: Eigenvector computation failed" << std::endl;
            return create_matrix(0, 0);
        }
        std::cout << "Eigenvectors matrix dimensions: " << eigenvectors.rows << " x " << eigenvectors.cols << std::endl;

        free_matrix(projected);
        projected.assign(matrix_multiply(data, eigenvectors));
        if (!projected.data) {
            std::cerr << "Error: Projection failed" << std::endl;
            return create_matrix(0, 0);
        }

        Matrix Vt = transpose(eigenvectors);
        if (!Vt.data) {
            std::cerr << "Error: Failed to transpose eigenvectors" << std::endl;
            free_matrix(projected);
            return create_matrix(0, 0);
        }
        Matrix reconstructed = matrix_multiply(projected, Vt);
        free_matrix(Vt);
        if (!reconstructed.data) {
            std::cerr << "Error: Reconstruction failed" << std::endl;
            free_matrix(projected);
            return create_matrix(0, 0);
        }
        std::cout << "Reconstructed matrix dimensions: " << reconstructed.rows << " x " << reconstructed.cols << std::endl;

        std::cout << "Variance of projected components:\n";
        for (int j = 0; j < n_components; ++j) {
            double mean = 0.0, variance = 0.0;
            for (int i = 0; i < projected.rows; ++i) {
                mean += projected.data[i * projected.cols + j];
            }
            mean /= projected.rows;
            for (int i = 0; i < projected.rows; ++i) {
                double diff = projected.data[i * projected.cols + j] - mean;
                variance += diff * diff;
            }
            variance /= projected.rows;
        }

        return reconstructed;
    }

    void centerData(Matrix& data) {
        if (!data.data) {
            std::cerr << "Error: Invalid data matrix for centering" << std::endl;
            return;
        }
        std::cout << "Centering data...\n";
        for (int j = 0; j < data.cols; ++j) {
            double mu = 0.0;
            for (int i = 0; i < data.rows; ++i) {
                mu += data.data[i * data.cols + j];
            }
            mu /= data.rows;
            mean.data[0 * mean.cols + j] = mu;
            for (int i = 0; i < data.rows; ++i) {
                data.data[i * data.cols + j] -= mu;
            }
        }

        std::cout << "Validating centering (mean of each column should be ~0):\n";
        for (int j = 0; j < data.cols; ++j) {
            double post_center_mean = 0.0;
            for (int i = 0; i < data.rows; ++i) {
                post_center_mean += data.data[i * data.cols + j];
            }
            post_center_mean /= data.rows;
            std::cout << "Column " << j + 1 << " mean after centering: " << post_center_mean << std::endl;
            if (std::abs(post_center_mean) > 1e-10) {
                std::cerr << "Warning: Column " << j + 1 << " is not centered properly (mean = "
                          << post_center_mean << ")" << std::endl;
            }
        }
    }

    const Matrix& getProjected() const { return projected; }
};

int main(int argc, char* argv[]) {
    int n_samples = 10000;
    int n_features = 20;
    double sparsity = 0.5;

    unsigned int seed = 42;
    unsigned int n_components = 10;

    if (argc > 1) {
        n_components = std::atoi(argv[1]);
    }
    if (argc > 2) {
        sparsity = std::atof(argv[2]);
    }
    if (argc > 3) {
        seed = std::atoi(argv[3]);
    }

    std::cout << "Generating random matrix: " << n_samples << " x " << n_features
              << ", sparsity = " << sparsity << ", seed = " << seed << std::endl;
    Matrix raw_data = generate_random_matrix(n_samples, n_features, sparsity, seed);
    if (!raw_data.data) {
        std::cerr << "Error: Failed to generate random matrix" << std::endl;
        return 1;
    }

    Matrix data;
    data.assign(scale_matrix(raw_data));
    free_matrix(raw_data);
    if (!data.data) {
        std::cerr << "Error: Failed to scale matrix" << std::endl;
        return 1;
    }
    std::cout << "Data matrix dimensions: " << data.rows << " x " << data.cols << std::endl;

    PCA pca(n_components, n_samples, n_features);

    auto start = std::chrono::high_resolution_clock::now();
    Matrix reconstructed;
    reconstructed.assign(pca.transform(data));
    if (!reconstructed.data) {
        std::cerr << "Error: Transform failed" << std::endl;
        free_matrix(data);
        return 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double rmse, frobenius;
    computeReconstructionError(data, reconstructed, rmse, frobenius);
    if (rmse < 0) {
        std::cerr << "Error: Failed to compute reconstruction error" << std::endl;
        free_matrix(data);
        free_matrix(reconstructed);
        return 1;
    }

    writeCSV("../results/pca/projected_data.csv", pca.getProjected());
    writeCSV("../results/pca/reconstructed_data.csv", reconstructed);

    std::ofstream result_file("../results/pca/results.csv");
    result_file << "execution_time_us,reconstruction_error_rmse,reconstruction_error_frobenius\n";
    result_file << duration.count() << "," << rmse << "," << frobenius << "\n";
    result_file.close();

    std::cout << "Execution time: " << duration.count() << " microseconds\n";
    std::cout << "Reconstruction error (RMSE): " << rmse << std::endl;
    std::cout << "Reconstruction error (Frobenius norm): " << frobenius << std::endl;
    std::cout << "Number of components used: " << n_components << std::endl;

    int non_zero_count = 0;
    for (int i = 0; i < data.rows; ++i) {
        for (int j = 0; j < data.cols; ++j) {
            if (data.data[i * data.cols + j] != 0.0) non_zero_count++;
        }
    }
    std::cout << "Non-zero elements: " << non_zero_count << ", sparsity: "
              << (1.0 - (double)non_zero_count / (data.rows * data.cols)) << std::endl;

    free_matrix(data);
    free_matrix(reconstructed);
    return 0;
}