#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <random>


struct DataPoint {
    __PROMISE__* features; 
    __PROMISE__ target;    
    int n_features;
};

struct Matrix {
    int rows, cols;
    __PROMISE__* data; // Row-major storage
};


Matrix create_matrix(int r, int c) {// Allocate matrix in row-major order
    Matrix A = {r, c, nullptr};
    if (r > 0 && c > 0) {
        A.data = new __PROMISE__[r * c]();
    }
    return A;
}

// Free matrix memory
void free_matrix(Matrix& A) {
    if (A.data) {
        delete[] A.data;
        A.data = nullptr;
    }
    A.rows = A.cols = 0;
}

// Access matrix element (i,j) with bounds checking, returns value
__PROMISE__ matrix_at(const Matrix& A, int i, int j) {
    if (i < 0 || i >= A.rows || j < 0 || j >= A.cols) {
        std::cerr << "Error: Matrix access out of bounds at (" << i << "," << j
                  << "), size (" << A.rows << "x" << A.cols << ")" << std::endl;
        throw std::out_of_range("Matrix access out of bounds");
    }
    return A.data[i * A.cols + j];
}

// Set matrix element (i,j) with bounds checking
void set_matrix_at(Matrix& A, int i, int j, __PROMISE__ value) {
    if (i < 0 || i >= A.rows || j < 0 || j >= A.cols) {
        std::cerr << "Error: Matrix access out of bounds at (" << i << "," << j
                  << "), size (" << A.rows << "x" << A.cols << ")" << std::endl;
        throw std::out_of_range("Matrix access out of bounds");
    }
    A.data[i * A.cols + j] = value;
}

Matrix transpose(const Matrix& A) {
    Matrix result = create_matrix(A.cols, A.rows);
    if (!result.data) return result;
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            set_matrix_at(result, j, i, matrix_at(A, i, j));
        }
    }
    return result;
}


Matrix matrix_multiply(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        std::cerr << "Matrix multiplication error: incompatible dimensions ("
                  << A.rows << "x" << A.cols << ") * (" << B.rows << "x" << B.cols << ")" << std::endl;
        return create_matrix(0, 0);
    }
    Matrix result = create_matrix(A.rows, B.cols);
    if (!result.data) return result;
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            __PROMISE__ sum = 0.0;
            for (int k = 0; k < A.cols; ++k) {
                sum += matrix_at(A, i, k) * matrix_at(B, k, j);
            }
            set_matrix_at(result, i, j, sum);
        }
    }
    return result;
}


Matrix pseudo_inverse(const Matrix& A) { // Pseudo-inverse using diagonal regularization
    int n = A.rows, m = A.cols;
    Matrix AtA = matrix_multiply(transpose(A), A);
    if (!AtA.data) return create_matrix(0, 0);
    Matrix result = create_matrix(m, n);
    if (!result.data) {
        free_matrix(AtA);
        return result;
    }

    double lambda = 1e-6;
    for (int i = 0; i < m; ++i) {
        __PROMISE__ diag = matrix_at(AtA, i, i) + lambda;
        set_matrix_at(AtA, i, i, diag);
    }

    if (n == m && n <= 50) {
        Matrix inv = create_matrix(m, m);
        if (!inv.data) {
            free_matrix(AtA);
            free_matrix(result);
            return create_matrix(0, 0);
        }
        for (int i = 0; i < m; ++i) {
            set_matrix_at(inv, i, i, 1.0 / matrix_at(AtA, i, i));
        }
        Matrix temp = matrix_multiply(inv, transpose(A));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                set_matrix_at(result, i, j, matrix_at(temp, i, j));
            }
        }
        free_matrix(inv);
        free_matrix(temp);
    }
    free_matrix(AtA);
    return result;
}


DataPoint* generate_random_data(int n_samples, int n_features, __PROMISE__ sparsity = 0.8, unsigned int seed = 42) {
    if (sparsity < 0.0 || sparsity > 1.0) {// Generate random dataset with sparsity control
        std::cerr << "Error: Sparsity must be between 0.0 and 1.0" << std::endl;
        return nullptr;
    }
    if (n_samples <= 0 || n_features <= 0) {
        std::cerr << "Error: Invalid dimensions n_samples=" << n_samples << ", n_features=" << n_features << std::endl;
        return nullptr;
    }

    DataPoint* data = new DataPoint[n_samples];
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> value_dis(-10.0, 10.0);
    std::uniform_real_distribution<> sparsity_dis(0.0, 1.0);

    for (int i = 0; i < n_samples; ++i) {
        data[i].features = new __PROMISE__[n_features]();
        data[i].n_features = n_features;
        data[i].target = 0.0;

        
        std::uniform_int_distribution<> idx_dis(0, n_features - 1);// Ensure at least one non-zero element per row
        int non_zero_idx = idx_dis(gen);
        data[i].features[non_zero_idx] = value_dis(gen);

        // Fill other elements with sparsity control
        for (int j = 0; j < n_features; ++j) {
            if (j == non_zero_idx) continue;
            if (sparsity_dis(gen) >= sparsity) {
                data[i].features[j] = value_dis(gen);
            }
        }
    }
    return data;
}

// Free DataPoint array
void free_data(DataPoint* data, int n_samples) {
    if (!data) return;
    for (int i = 0; i < n_samples; ++i) {
        delete[] data[i].features;
        data[i].features = nullptr;
    }
    delete[] data;
}


DataPoint* scale_features(DataPoint* data, int n_samples, int n_features) {
    // Scale features (standardization, handling sparse data)
    if (!data || n_samples <= 0 || n_features <= 0) return nullptr;
    DataPoint* scaled_data = new DataPoint[n_samples];
    __PROMISE__* means = new __PROMISE__[n_features]();
    __PROMISE__* stds = new __PROMISE__[n_features]();
    int* counts = new int[n_features]();

    // Initialize scaled_data
    for (int i = 0; i < n_samples; ++i) {
        scaled_data[i].features = new __PROMISE__[n_features]();
        scaled_data[i].n_features = n_features;
        scaled_data[i].target = data[i].target;
    }

    // Compute means (only for non-zero elements)
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            scaled_data[i].features[j] = data[i].features[j];
            if (data[i].features[j] != 0.0) {
                means[j] += data[i].features[j];
                counts[j]++;
            }
        }
    }
    for (int j = 0; j < n_features; ++j) {
        double temp = 1.0;
        means[j] = counts[j] > 0 ? means[j] / counts[j] : temp;
    }

    // Compute standard deviations
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            if (data[i].features[j] != 0.0) {
                __PROMISE__ diff = data[i].features[j] - means[j];
                stds[j] += diff * diff;
            }
        }
    }
    for (int j = 0; j < n_features; ++j) {
        double temp = 1.0;
        stds[j] = counts[j] > 0 ? sqrt(stds[j] / counts[j]) : temp;
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }

    // Scale features (preserve zeros)
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            if (data[i].features[j] != 0.0) {
                scaled_data[i].features[j] = (data[i].features[j] - means[j]) / stds[j];
            }
        }
    }

    delete[] means;
    delete[] stds;
    delete[] counts;
    return scaled_data;
}

// Convert DataPoint array to Matrix
Matrix data_to_matrix(DataPoint* data, int n_samples, int n_features) {
    Matrix X = create_matrix(n_samples, n_features);
    if (!X.data) return X;
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            set_matrix_at(X, i, j, data[i].features[j]);
        }
    }
    return X;
}

class Nystrom {
private:
    int n_components;
    Matrix C;      // n_samples x n_components
    Matrix W_inv;  // n_components x n_components
    int* sampled_indices;

public:
    Nystrom(int k = 5) : n_components(k), sampled_indices(nullptr) {
        C = create_matrix(0, 0);
        W_inv = create_matrix(0, 0);
    }
    ~Nystrom() {
        free_matrix(C);
        free_matrix(W_inv);
        delete[] sampled_indices;
        sampled_indices = nullptr;
    }

    void fit(const Matrix& X) {
        if (X.rows <= 0 || X.cols <= 0 || !X.data) {
            std::cerr << "Error: Invalid input matrix for fit" << std::endl;
            return;
        }
        int n_samples = X.rows;
        int n_features = X.cols;
        if (n_components > n_features) n_components = n_features;

        // Free previous allocations
        free_matrix(C);
        free_matrix(W_inv);
        delete[] sampled_indices;

        // Randomly select n_components indices
        int* indices = new int[n_features];
        for (int i = 0; i < n_features; ++i) indices[i] = i;
        std::mt19937 gen(42);
        for (int i = n_features - 1; i > 0; --i) {
            std::uniform_int_distribution<> dis(0, i);
            int j = dis(gen);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        sampled_indices = new int[n_components];
        for (int i = 0; i < n_components; ++i) {
            sampled_indices[i] = indices[i];
        }
        delete[] indices;

        // Build C matrix
        C = create_matrix(n_samples, n_components);
        if (!C.data) {
            std::cerr << "Error: Failed to allocate C matrix" << std::endl;
            return;
        }
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < n_components; ++j) {
                set_matrix_at(C, i, j, matrix_at(X, i, sampled_indices[j]));
            }
        }

        // Compute W = C^T * C and its pseudo-inverse
        Matrix W = matrix_multiply(transpose(C), C);
        if (!W.data) {
            std::cerr << "Error: Failed to compute W matrix" << std::endl;
            free_matrix(C);
            return;
        }
        W_inv = pseudo_inverse(W);
        free_matrix(W);
    }

    Matrix transform(const Matrix& X) {
        if (!C.data || !W_inv.data) {
            std::cerr << "Error: Nystrom not fitted properly" << std::endl;
            return create_matrix(0, 0);
        }
        // Compute X_reduced = C * W_inv
        Matrix result = matrix_multiply(C, W_inv); // (n_samples x n_components) * (n_components x n_components) = n_samples x n_components
        if (!result.data) {
            std::cerr << "Error: Failed to compute C * W_inv" << std::endl;
            return create_matrix(0, 0);
        }
        return result;
    }

    Matrix reconstruct(const Matrix& X) {
        if (!C.data || !W_inv.data) {
            std::cerr << "Error: Nystrom not fitted properly" << std::endl;
            return create_matrix(0, 0);
        }
        // Compute X_reconstructed = C * W_inv * C^T * X
        Matrix Ct = transpose(C); // n_components x n_samples
        Matrix temp1 = matrix_multiply(Ct, X); // (n_components x n_samples) * (n_samples x n_features) = n_components x n_features
        free_matrix(Ct);
        if (!temp1.data) {
            std::cerr << "Error: Failed to compute C^T * X" << std::endl;
            return create_matrix(0, 0);
        }
        Matrix temp2 = matrix_multiply(W_inv, temp1); // (n_components x n_components) * (n_components x n_features) = n_components x n_features
        free_matrix(temp1);
        if (!temp2.data) {
            std::cerr << "Error: Failed to compute W_inv * (C^T * X)" << std::endl;
            return create_matrix(0, 0);
        }
        Matrix result = matrix_multiply(C, temp2); // (n_samples x n_components) * (n_components x n_features) = n_samples x n_features
        free_matrix(temp2);
        if (!result.data) {
            std::cerr << "Error: Failed to compute C * (W_inv * C^T * X)" << std::endl;
            return create_matrix(0, 0);
        }
        return result;
    }
};

// Compute reconstruction error (Frobenius norm)
double compute_reconstruction_error(const Matrix& original, const Matrix& reconstructed) {
    if (original.rows != reconstructed.rows || original.cols != reconstructed.cols || !original.data || !reconstructed.data) {
        std::cerr << "Dimension mismatch or invalid data in reconstruction error calculation: original ("
                  << original.rows << "x" << original.cols << "), reconstructed ("
                  << reconstructed.rows << "x" << reconstructed.cols << ")" << std::endl;
        return -1.0;
    }
    double error = 0.0;
    for (int i = 0; i < original.rows; ++i) {
        for (int j = 0; j < original.cols; ++j) {
            double diff = matrix_at(original, i, j) - matrix_at(reconstructed, i, j);
            error += diff * diff;
        }
    }
    return sqrt(error);
}



int main() {
    try {
        int n_samples = 9999;
        int n_features = 20;
        int n_components = std::min(10, n_features);
        double sparsity = 1;

        DataPoint* raw_data = generate_random_data(n_samples, n_features, sparsity);
        if (!raw_data) {
            std::cerr << "Error: Data generation failed" << std::endl;
            return 1;
        }
        DataPoint* data = scale_features(raw_data, n_samples, n_features);
        if (!data) {
            std::cerr << "Error: Feature scaling failed" << std::endl;
            free_data(raw_data, n_samples);
            return 1;
        }

        Matrix X = data_to_matrix(data, n_samples, n_features);
        if (!X.data) {
            std::cerr << "Error: Failed to create input matrix" << std::endl;
            free_data(raw_data, n_samples);
            free_data(data, n_samples);
            return 1;
        }

        Nystrom nystrom(n_components);

        nystrom.fit(X);
        Matrix X_reduced = nystrom.transform(X);
        if (!X_reduced.data) {
            std::cerr << "Error: Transform failed" << std::endl;
            free_data(raw_data, n_samples);
            free_data(data, n_samples);
            free_matrix(X);
            return 1;
        }
        Matrix X_reconstructed = nystrom.reconstruct(X);
        if (!X_reconstructed.data) {
            std::cerr << "Error: Reconstruction failed" << std::endl;
            free_data(raw_data, n_samples);
            free_data(data, n_samples);
            free_matrix(X);
            free_matrix(X_reduced);
            return 1;
        }

        double recon_error = compute_reconstruction_error(X, X_reconstructed);
        std::cout << "Reconstruction Error (Frobenius norm): " << recon_error << std::endl;

        int check_len = X_reduced.rows * X_reduced.cols;
        double check_x[check_len]; // add for check
        for (int i=0; i< check_len; i++){
            check_x[i]= X_reduced.data[i];
        }
        // PROMISE_CHECK_VAR(recon_error);
        PROMISE_CHECK_ARRAY(check_x, check_len);
    
    
        free_data(raw_data, n_samples);
        free_data(data, n_samples);
        free_matrix(X);
        free_matrix(X_reduced);
        free_matrix(X_reconstructed);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}