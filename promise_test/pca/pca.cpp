#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>  

struct Matrix;

Matrix create_matrix(int r, int c);


void free_matrix(Matrix& A);


struct Matrix {
    int rows, cols;
    __PROMISE__* data; // Row-major storage

    Matrix() : rows(0), cols(0), data(nullptr) {}

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(nullptr) {
        if (other.data && rows > 0 && cols > 0) {
            data = new __PROMISE__[rows * cols];
            std::memcpy(data, other.data, rows * cols * sizeof(__PROMISE__));
        }
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            free_matrix(*this);
            rows = other.rows;
            cols = other.cols;
            if (other.data && rows > 0 && cols > 0) {
                data = new __PROMISE__[rows * cols];
                std::memcpy(data, other.data, rows * cols * sizeof(__PROMISE__));
            }
        }
        return *this;
    }

    Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data) {
        other.rows = 0;
        other.cols = 0;
        other.data = nullptr;
    }

    
    Matrix& operator=(Matrix&& other) noexcept { 
        if (this != &other) {
            free_matrix(*this);
            rows = other.rows;
            cols = other.cols;
            data = other.data;
            other.rows = 0;
            other.cols = 0;
            other.data = nullptr;
        }
        return *this;
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
            A.data = new __PROMISE__[r * c]();
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

__PROMISE__& matrix_at(Matrix& A, int i, int j) {
    if (!A.data || i < 0 || i >= A.rows || j < 0 || j >= A.cols) {
        std::cerr << "Error: Matrix access out of bounds at (" << i << "," << j
                  << "), size (" << A.rows << "x" << A.cols << ")" << std::endl;
        throw std::out_of_range("Matrix access out of bounds");
    }
    return A.data[i * A.cols + j];
}

const __PROMISE__ matrix_at(const Matrix& A, int i, int j) {
    if (!A.data || i < 0 || i >= A.rows || j < 0 || j >= A.cols) {
        std::cerr << "Error: Matrix access out of bounds at (" << i << "," << j
                  << "), size (" << A.rows << "x" << A.cols << ")" << std::endl;
        throw std::out_of_range("Matrix access out of bounds");
    }
    return A.data[i * A.cols + j];
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
            __PROMISE__ sum = 0.0;
            for (int k = 0; k < A.cols; ++k) {
                sum += matrix_at(A, i, k) * matrix_at(B, k, j);
            }
            matrix_at(result, i, j) = sum;
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
            matrix_at(result, j, i) = matrix_at(A, i, j);
        }
    }
    return result;
}


Matrix generate_random_matrix(int n_samples, int n_features, __PROMISE__ sparsity = 0.8, unsigned int seed = 42) {
    if (sparsity < 0.0 || sparsity > 1.0) { // Generate random matrix with sparsity control
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

    for (int i = 0; i < n_samples; ++i) { // Ensure at least one non-zero element per row
        std::uniform_int_distribution<> idx_dis(0, n_features - 1);
        int non_zero_idx = idx_dis(gen);
        matrix_at(data, i, non_zero_idx) = value_dis(gen);

        for (int j = 0; j < n_features; ++j) {  // Fill other elements with sparsity control
            if (j == non_zero_idx) continue;
            if (sparsity_dis(gen) >= sparsity) {
                matrix_at(data, i, j) = value_dis(gen);
            }
        }
    }

    return data;
}


Matrix scale_matrix(const Matrix& input) { // Scale matrix (standardize non-zero elements)
    if (!input.data || input.rows <= 0 || input.cols <= 0) {
        std::cerr << "Error: Invalid input matrix for scaling" << std::endl;
        return create_matrix(0, 0);
    }

    Matrix scaled = create_matrix(input.rows, input.cols);
    if (!scaled.data) return scaled;

    __PROMISE__* means = new __PROMISE__[input.cols]();
    __PROMISE__* stds = new __PROMISE__[input.cols]();
    int* counts = new int[input.cols]();

    
    for (int i = 0; i < input.rows; ++i) { // Copy input to scaled and compute means for non-zero elements
        for (int j = 0; j < input.cols; ++j) {
            __PROMISE__ val = matrix_at(input, i, j);
            matrix_at(scaled, i, j) = val;
            if (val != 0.0) {
                means[j] += val;
                counts[j]++;
            }
        }
    }
    for (int j = 0; j < input.cols; ++j) {
        means[j] = counts[j] > 0 ? means[j] / counts[j] : 0.0;
    }

    // Compute standard deviations
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            __PROMISE__ val = matrix_at(input, i, j);
            if (val != 0.0) {
                __PROMISE__ diff = val - means[j];
                stds[j] += diff * diff;
            }
        }
    }
    for (int j = 0; j < input.cols; ++j) {
        stds[j] = counts[j] > 0 ? sqrt(stds[j] / counts[j]) : 1.0;
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }

    // Scale non-zero elements
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            __PROMISE__ val = matrix_at(input, i, j);
            if (val != 0.0) {
                matrix_at(scaled, i, j) = (val - means[j]) / stds[j];
            }
        }
    }

    delete[] means;
    delete[] stds;
    delete[] counts;
    return scaled;
}

// Compute reconstruction error (RMSE and Frobenius norm)
void computeReconstructionError(const Matrix& original, const Matrix& reconstructed, __PROMISE__& rmse, __PROMISE__& frobenius) {
    if (!original.data || !reconstructed.data || original.rows != reconstructed.rows || original.cols != reconstructed.cols) {
        std::cerr << "Dimension mismatch or invalid data in reconstruction error calculation: original ("
                  << original.rows << "x" << original.cols << "), reconstructed ("
                  << reconstructed.rows << "x" << reconstructed.cols << ")" << std::endl;
        rmse = frobenius = -1.0;
        return;
    }
    __PROMISE__ error = 0.0;
    for (int i = 0; i < original.rows; ++i) {
        for (int j = 0; j < original.cols; ++j) {
            __PROMISE__ diff = matrix_at(original, i, j) - matrix_at(reconstructed, i, j);
            error += diff * diff;
        }
    }
    frobenius = sqrt(error);
    rmse = sqrt(error / (original.rows * original.cols));
}

class PCA {
private:
    int n_components;
    Matrix mean;        // 1 x cols
    Matrix projected;   // rows x n_components
    Matrix eigenvectors; // cols x n_components
    Matrix eigenvalues; // 1 x n_components


    void power_iteration(const Matrix& A, __PROMISE__& eigenvalue, Matrix& eigenvector, int max_iter = 200) {
        if (!A.data || A.rows != A.cols) {     // Power iteration for largest eigenvalue and eigenvector
            std::cerr << "Error: Invalid matrix for power iteration ("
                      << A.rows << "x" << A.cols << ")" << std::endl;
            eigenvalue = 0.0;
            eigenvector = create_matrix(0, 0);
            return;
        }
        int n = A.rows;
        eigenvector = create_matrix(n, 1);
        if (!eigenvector.data) {
            std::cerr << "Error: Failed to allocate eigenvector" << std::endl;
            eigenvalue = 0.0;
            return;
        }

        // Initialize random vector
        std::mt19937 gen(42); // Fixed seed
        std::uniform_real_distribution<> dis(0.0, 1.0);
        __PROMISE__ norm = 0.0;
        for (int i = 0; i < n; ++i) {
            __PROMISE__ val = dis(gen);
            matrix_at(eigenvector, i, 0) = val;
            norm += val * val;
        }
        norm = sqrt(norm);
        if (norm < 1e-10) {
            std::cerr << "Error: Zero norm in eigenvector initialization" << std::endl;
            eigenvalue = 0.0;
            free_matrix(eigenvector);
            return;
        }
        for (int i = 0; i < n; ++i) {
            matrix_at(eigenvector, i, 0) /= norm;
        }

        // Power iteration
        Matrix temp = create_matrix(n, 1);
        if (!temp.data) {
            std::cerr << "Error: Failed to allocate temp vector in power iteration" << std::endl;
            eigenvalue = 0.0;
            free_matrix(eigenvector);
            return;
        }
        __PROMISE__ prev_eigenvalue = 0.0;
        for (int iter = 0; iter < max_iter; ++iter) {
            // temp = A * eigenvector
            for (int i = 0; i < n; ++i) {
                __PROMISE__ sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    sum += matrix_at(A, i, j) * matrix_at(eigenvector, j, 0);
                }
                matrix_at(temp, i, 0) = sum;
            }
            // Compute eigenvalue (Rayleigh quotient)
            eigenvalue = 0.0;
            for (int i = 0; i < n; ++i) {
                eigenvalue += matrix_at(eigenvector, i, 0) * matrix_at(temp, i, 0);
            }
            // Normalize
            norm = 0.0;
            for (int i = 0; i < n; ++i) {
                norm += matrix_at(temp, i, 0) * matrix_at(temp, i, 0);
            }
            norm = sqrt(norm);
            if (norm < 1e-10) {
                std::cerr << "Error: Zero norm in power iteration at iter " << iter << std::endl;
                eigenvalue = 0.0;
                free_matrix(temp);
                free_matrix(eigenvector);
                return;
            }
            for (int i = 0; i < n; ++i) {
                matrix_at(eigenvector, i, 0) = matrix_at(temp, i, 0) / norm;
            }
            // Check convergence
            if (iter > 0 && abs(eigenvalue - prev_eigenvalue) < 1e-8) {
                break;
            }
            prev_eigenvalue = eigenvalue;
        }
        free_matrix(temp);
    }

    
    void computeEigenvectors(Matrix& data, int rows, int cols) { // Deflation to compute subsequent eigenvectors
        if (!data.data) {
            std::cerr << "Error: Invalid data matrix for eigenvector computation" << std::endl;
            return;
        }
        // Compute covariance matrix: (1/rows) * X^T * X
        Matrix Xt = transpose(data); // cols x rows
        Matrix cov = matrix_multiply(Xt, data); // cols x cols
        free_matrix(Xt);
        if (!cov.data) {
            std::cerr << "Error: Failed to compute covariance matrix" << std::endl;
            return;
        }
        for (int i = 0; i < cov.rows; ++i) {
            for (int j = 0; j < cov.cols; ++j) {
                matrix_at(cov, i, j) /= rows;
            }
        }

        // Compute n_components eigenvectors using power iteration with deflation
        eigenvectors = create_matrix(cols, n_components); // d x k (20 x 3)
        eigenvalues = create_matrix(1, n_components);
        if (!eigenvectors.data || !eigenvalues.data) {
            std::cerr << "Error: Failed to allocate eigenvectors or eigenvalues" << std::endl;
            free_matrix(cov);
            return;
        }
        Matrix A = cov; // Working copy
        for (int k = 0; k < n_components; ++k) {
            __PROMISE__ eigenvalue;
            Matrix eigenvector;
            power_iteration(A, eigenvalue, eigenvector);
            if (!eigenvector.data) {
                std::cerr << "Error: Power iteration failed for component " << k + 1 << std::endl;
                free_matrix(A);
                free_matrix(cov);
                return;
            }
            matrix_at(eigenvalues, 0, k) = eigenvalue;
            // Store eigenvector in column k
            for (int i = 0; i < cols; ++i) {
                matrix_at(eigenvectors, i, k) = matrix_at(eigenvector, i, 0);
            }

            // Deflation: A = A - eigenvalue * v * v^T
            Matrix vvt = matrix_multiply(eigenvector, transpose(eigenvector)); // n x n
            if (!vvt.data) {
                std::cerr << "Error: Failed to compute v*v^T for deflation" << std::endl;
                free_matrix(eigenvector);
                free_matrix(A);
                free_matrix(cov);
                return;
            }
            for (int i = 0; i < A.rows; ++i) {
                for (int j = 0; j < A.cols; ++j) {
                    matrix_at(A, i, j) -= eigenvalue * matrix_at(vvt, i, j);
                }
            }
            free_matrix(vvt);
            free_matrix(eigenvector);

            // std::cout << "Eigenvalue " << k + 1 << ": " << eigenvalue << std::endl;
        }
        free_matrix(A);
        free_matrix(cov);

        for (int k = 0; k < n_components; ++k) {
            for (int m = 0; m < k; ++m) {
                __PROMISE__ dot = 0.0;
                for (int i = 0; i < cols; ++i) {
                    dot += matrix_at(eigenvectors, i, k) * matrix_at(eigenvectors, i, m);
                }
                // std::cout << "Dot product of eigenvectors " << m + 1 << " and " << k + 1 << ": " << dot << std::endl;
            }
        }
    }

public:
    PCA(int n_components, int rows, int cols) : n_components(n_components) {
        if (n_components <= 0 || n_components > cols || n_components > rows) {
            std::cerr << "Error: Invalid number of components: " << n_components
                      << " (must be positive and <= min(rows, cols) = " << std::min(rows, cols) << ")" << std::endl;
            exit(1);
        }
        mean = create_matrix(1, cols);
        projected = create_matrix(0, 0);
        eigenvectors = create_matrix(0, 0);
        eigenvalues = create_matrix(0, 0);
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

        // Project data: X * V (rows x cols) * (cols x n_components) = rows x n_components
        projected = matrix_multiply(data, eigenvectors); // (9999 x 20) * (20 x 3) = 9999 x 3
        if (!projected.data) {
            std::cerr << "Error: Projection failed" << std::endl;
            return create_matrix(0, 0);
        }
        std::cout << "Projected matrix dimensions: " << projected.rows << " x " << projected.cols << std::endl;

        // Reconstruct: X_projected * V^T (rows x n_components) * (n_components x cols) = rows x cols
        Matrix Vt = transpose(eigenvectors); // 3 x 20
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
            __PROMISE__ mean = 0.0, variance = 0.0;
            for (int i = 0; i < projected.rows; ++i) {
                mean += matrix_at(projected, i, j);
            }
            mean /= projected.rows;
            for (int i = 0; i < projected.rows; ++i) {
                __PROMISE__ diff = matrix_at(projected, i, j) - mean;
                variance += diff * diff;
            }
            variance /= projected.rows;
            // std::cout << "Component " << j + 1 << ": " << variance
            //          << " (Eigenvalue: " << matrix_at(eigenvalues, 0, j) << ")" << std::endl;
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
            __PROMISE__ mu = 0.0;
            for (int i = 0; i < data.rows; ++i) {
                mu += matrix_at(data, i, j);
            }
            mu /= data.rows;
            matrix_at(mean, 0, j) = mu;
            for (int i = 0; i < data.rows; ++i) {
                matrix_at(data, i, j) -= mu;
            }
        }

        std::cout << "Validating centering (mean of each column should be ~0):\n";
        for (int j = 0; j < data.cols; ++j) {
            __PROMISE__ post_center_mean = 0.0;
            for (int i = 0; i < data.rows; ++i) {
                post_center_mean += matrix_at(data, i, j);
            }
            post_center_mean /= data.rows;
            // std::cout << "Column " << j + 1 << " mean after centering: " << post_center_mean << std::endl;
            if (abs(post_center_mean) > 1e-10) {
                std::cerr << "Warning: Column " << j + 1 << " is not centered properly (mean = "
                          << post_center_mean << ")" << std::endl;
            }
        }
    }

    const Matrix& getProjected() const { return projected; }
};

int main(int argc, char* argv[]) {
    int n_samples = 9999;
    int n_features = 20;
    __PROMISE__ sparsity = 1;
    unsigned int seed = 42;
    unsigned int n_components = 10;

    // if (argc > 1) {
    //     n_components = std::atoi(argv[1]);
    // }
    // if (argc > 2) {
    //     sparsity = std::atof(argv[2]);
    // }
    // if (argc > 3) {
    //     seed = std::atoi(argv[3]);
    // }

    std::cout << "Generating random matrix: " << n_samples << " x " << n_features
              << ", sparsity = " << sparsity << ", seed = " << seed << std::endl;
    Matrix raw_data = generate_random_matrix(n_samples, n_features, sparsity, seed);
    if (!raw_data.data) {
        std::cerr << "Error: Failed to generate random matrix" << std::endl;
        return 1;
    }

    Matrix data = scale_matrix(raw_data);
    free_matrix(raw_data);
    if (!data.data) {
        std::cerr << "Error: Failed to scale matrix" << std::endl;
        return 1;
    }
    std::cout << "Data matrix dimensions: " << data.rows << " x " << data.cols << std::endl;

    PCA pca(n_components, n_samples, n_features);

    Matrix reconstructed = pca.transform(data);
    if (!reconstructed.data) {
        std::cerr << "Error: Transform failed" << std::endl;
        free_matrix(data);
        return 1;
    }

    double check_x[reconstructed.rows * reconstructed.cols]; // add for check
    for (int i=0; i<reconstructed.rows * reconstructed.cols; i++){
        check_x[i] = reconstructed.data[i];
    }


    PROMISE_CHECK_ARRAY(check_x, reconstructed.rows * reconstructed.cols)
    __PROMISE__ rmse, frobenius;
    computeReconstructionError(data, reconstructed, rmse, frobenius);
    if (rmse < 0) {
        std::cerr << "Error: Failed to compute reconstruction error" << std::endl;
        free_matrix(data);
        free_matrix(reconstructed);
        return 1;
    }

    std::cout << "Reconstruction error (RMSE): " << rmse << std::endl;
    std::cout << "Reconstruction error (Frobenius norm): " << frobenius << std::endl;
    std::cout << "Number of components used: " << n_components << std::endl;

    int non_zero_count = 0;
    for (int i = 0; i < data.rows; ++i) {
        for (int j = 0; j < data.cols; ++j) {
            if (matrix_at(data, i, j) != 0.0) non_zero_count++;
        }
    }
    std::cout << "Non-zero elements: " << non_zero_count << ", sparsity: "
              << (1.0 - (__PROMISE__)non_zero_count / (data.rows * data.cols)) << std::endl;

    free_matrix(data);
    free_matrix(reconstructed);
    return 0;
}