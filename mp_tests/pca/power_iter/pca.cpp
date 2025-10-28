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
    __PROMISE__* data;

    Matrix() : rows(0), cols(0), data(nullptr) {}

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(nullptr) {
        if (other.data && rows > 0 && cols > 0) {
            data = new __PROMISE__[rows * cols];
            std::memcpy(data, other.data, rows * cols * sizeof(__PROMISE__));
        }
    }

    Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(other.data) {
        other.rows = 0;
        other.cols = 0;
        other.data = nullptr;
    }

    // Assignment function for copy assignment
    void assign(const Matrix& other) {
        if (this != &other) {
            if (data) {
                delete[] data;
            }
            rows = other.rows;
            cols = other.cols;
            data = nullptr;
            if (other.data && rows > 0 && cols > 0) {
                data = new __PROMISE__[rows * cols];
                std::memcpy(data, other.data, rows * cols * sizeof(__PROMISE__));
            }
        }
    }

    // Assignment function for move assignment
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

Matrix generate_random_matrix(int n_samples, int n_features, __PROMISE__ sparsity = 0.8, unsigned int seed = 42) {
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

    __PROMISE__* means = new __PROMISE__[input.cols]();
    __PROMISE__* stds = new __PROMISE__[input.cols]();
    int* counts = new int[input.cols]();

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            __PROMISE__ val = input.data[i * input.cols + j];
            scaled.data[i * scaled.cols + j] = val;
            if (val != 0.0) {
                means[j] += val;
                counts[j]++;
            }
        }
    }
    for (int j = 0; j < input.cols; ++j) {
        if (counts[j] > 0 ){
            means[j] = means[j] / counts[j];
        }
        else{
            means[j] = 0.0;
        }
        
        //means[j] = counts[j] > 0 ? means[j] / counts[j] : 0.0;
    }

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            __PROMISE__ val = input.data[i * input.cols + j];
            if (val != 0.0) {
                __PROMISE__ diff = val - means[j];
                stds[j] += diff * diff;
            }
        }
    }
    for (int j = 0; j < input.cols; ++j) {
        if (counts[j] > 0){
            stds[j] = sqrt(stds[j] / counts[j]);
        }
        else{
            stds[j] = 1.0;
        }
        // stds[j] = counts[j] > 0 ? sqrt(stds[j] / counts[j]) : 1.0;
        if (stds[j] < 1e-9) stds[j] = 1e-9;
    }

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            __PROMISE__ val = input.data[i * input.cols + j];
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


class PCA {
private:
    int n_components;
    Matrix mean;
    Matrix projected;
    Matrix eigenvectors;
    Matrix eigenvalues;

    void power_iteration(const Matrix A, __PROMISE__ eigenvalue, Matrix eigenvector, int max_iter = 200) {
        if (!A.data || A.rows != A.cols) {
            std::cerr << "Error: Invalid matrix for power iteration ("
                      << A.rows << "x" << A.cols << ")" << std::endl;
            eigenvalue = 0.0;
            free_matrix(eigenvector);
            eigenvector.assign(create_matrix(0, 0));
            return;
        }
        int n = A.rows;
        free_matrix(eigenvector);
        eigenvector.assign(create_matrix(n, 1));
        if (!eigenvector.data) {
            std::cerr << "Error: Failed to allocate eigenvector" << std::endl;
            eigenvalue = 0.0;
            return;
        }

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dis(0.0, 1.0);
        __PROMISE__ norm = 0.0;
        for (int i = 0; i < n; ++i) {
            __PROMISE__ val = dis(gen);
            eigenvector.data[i * eigenvector.cols + 0] = val;
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
            eigenvector.data[i * eigenvector.cols + 0] /= norm;
        }

        Matrix temp = create_matrix(n, 1);
        if (!temp.data) {
            std::cerr << "Error: Failed to allocate temp vector in power iteration" << std::endl;
            eigenvalue = 0.0;
            free_matrix(temp);
            free_matrix(eigenvector);
            return;
        }
        __PROMISE__ prev_eigenvalue = 0.0;
        for (int iter = 0; iter < max_iter; ++iter) {
            for (int i = 0; i < n; ++i) {
                __PROMISE__ sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    sum += A.data[i * A.cols + j] * eigenvector.data[j * eigenvector.cols + 0];
                }
                temp.data[i * temp.cols + 0] = sum;
            }
            eigenvalue = 0.0;
            for (int i = 0; i < n; ++i) {
                eigenvalue += eigenvector.data[i * eigenvector.cols + 0] * temp.data[i * temp.cols + 0];
            }
            norm = 0.0;
            for (int i = 0; i < n; ++i) {
                norm += temp.data[i * temp.cols + 0] * temp.data[i * temp.cols + 0];
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
                eigenvector.data[i * eigenvector.cols + 0] = temp.data[i * temp.cols + 0] / norm;
            }
            if (iter > 0 && abs(eigenvalue - prev_eigenvalue) < 1e-8) {
                break;
            }
            prev_eigenvalue = eigenvalue;
        }
        free_matrix(temp);
    }

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

        free_matrix(eigenvectors);
        eigenvectors.assign(create_matrix(cols, n_components));
        free_matrix(eigenvalues);
        eigenvalues.assign(create_matrix(1, n_components));
        if (!eigenvectors.data || !eigenvalues.data) {
            std::cerr << "Error: Failed to allocate eigenvectors or eigenvalues" << std::endl;
            free_matrix(cov);
            return;
        }
        Matrix A;
        A.assign(cov);
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
            eigenvalues.data[0 * eigenvalues.cols + k] = eigenvalue;
            for (int i = 0; i < cols; ++i) {
                eigenvectors.data[i * eigenvectors.cols + k] = eigenvector.data[i * eigenvector.cols + 0];
            }

            Matrix vvt = matrix_multiply(eigenvector, transpose(eigenvector));
            if (!vvt.data) {
                std::cerr << "Error: Failed to compute v*v^T for deflation" << std::endl;
                free_matrix(eigenvector);
                free_matrix(A);
                free_matrix(cov);
                return;
            }
            for (int i = 0; i < A.rows; ++i) {
                for (int j = 0; j < A.cols; ++j) {
                    A.data[i * A.cols + j] -= eigenvalue * vvt.data[i * vvt.cols + j];
                }
            }
            free_matrix(vvt);
            free_matrix(eigenvector);

            std::cout << "Eigenvalue " << k + 1 << ": " << eigenvalue << std::endl;
        }
        free_matrix(A);
        free_matrix(cov);

        for (int k = 0; k < n_components; ++k) {
            for (int m = 0; m < k; ++m) {
                __PROMISE__ dot = 0.0;
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
            __PROMISE__ mean = 0.0, variance = 0.0;
            for (int i = 0; i < projected.rows; ++i) {
                mean += projected.data[i * projected.cols + j];
            }
            mean /= projected.rows;
            for (int i = 0; i < projected.rows; ++i) {
                __PROMISE__ diff = projected.data[i * projected.cols + j] - mean;
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
            __PROMISE__ mu = 0.0;
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
            __PROMISE__ post_center_mean = 0.0;
            for (int i = 0; i < data.rows; ++i) {
                post_center_mean += data.data[i * data.cols + j];
            }
            post_center_mean /= data.rows;
            std::cout << "Column " << j + 1 << " mean after centering: " << post_center_mean << std::endl;
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
    __PROMISE__ sparsity = 0.1;
    unsigned int seed = 42;
    unsigned int n_components = 5;

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

    Matrix reconstructed;
    reconstructed.assign(pca.transform(data));
    if (!reconstructed.data) {
        std::cerr << "Error: Transform failed" << std::endl;
        free_matrix(data);
        return 1;
    }


    __PROMISE__  check_x[reconstructed.rows * reconstructed.cols]; // add for check
    for (int i=0; i<reconstructed.rows * reconstructed.cols; i++){
        check_x[i] = reconstructed.data[i];
    }


    PROMISE_CHECK_ARRAY(check_x, reconstructed.rows * reconstructed.cols);

    int non_zero_count = 0;
    for (int i = 0; i < data.rows; ++i) {
        for (int j = 0; j < data.cols; ++j) {
            if (data.data[i * data.cols + j] != 0.0) non_zero_count++;
        }
    }
    std::cout << "Non-zero elements: " << non_zero_count << ", sparsity: "
              << (1.0 - (__PROMISE__)non_zero_count / (data.rows * data.cols)) << std::endl;

    free_matrix(data);
    free_matrix(reconstructed);
    return 0;
}