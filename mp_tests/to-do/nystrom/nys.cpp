#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iomanip>
#include <string>
#include <chrono>
#include <random>
#include <cstring>

class Matrix {
public:
    __PROMISE__* data;
    int rows, cols;

    Matrix() : rows(0), cols(0), data(nullptr) {}

    Matrix(int r, int c) : rows(r), cols(c) {
        data = new __PROMISE__[r * c]();
    }

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        data = new __PROMISE__[rows * cols];
        std::memcpy(data, other.data, rows * cols * sizeof(__PROMISE__));
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            delete[] data;
            rows = other.rows;
            cols = other.cols;
            data = new __PROMISE__[rows * cols];
            std::memcpy(data, other.data, rows * cols * sizeof(__PROMISE__));
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
            delete[] data;
            rows = other.rows;
            cols = other.cols;
            data = other.data;
            other.rows = 0;
            other.cols = 0;
            other.data = nullptr;
        }
        return *this;
    }

    ~Matrix() {
        delete[] data;
    }

    __PROMISE__ get(int i, int j) const {

        return data[i * cols + j];
    }

    void set(int i, int j, __PROMISE__ value) {

        data[i * cols + j] = value;
    }

    void resize(int r, int c) {
        delete[] data;
        rows = r;
        cols = c;
        data = new __PROMISE__[r * c]();
    }
};

Matrix read_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    int max_rows = 1000; // Initial capacity
    int max_cols = 11;
    __PROMISE__** temp_data = new __PROMISE__*[max_rows];
    for (int i = 0; i < max_rows; ++i) {
        temp_data[i] = new __PROMISE__[max_cols];
    }
    int line_count = 0;
    std::string line;
    bool first_line = true;
    const char* expected_headers[] = {"age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"};
    int expected_cols = 10;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string* tokens = new std::string[max_cols];
        int token_count = 0;
        std::string val;
        while (std::getline(ss, val, ',')) {
            if (token_count < max_cols) {
                tokens[token_count++] = val;
            }
        }

        if (first_line) {
            if (token_count < expected_cols + 1) {
                delete[] tokens;
                for (int i = 0; i < max_rows; ++i) delete[] temp_data[i];
                delete[] temp_data;
                throw std::runtime_error("Invalid header: too few columns");
            }
            for (int i = 0; i < expected_cols; ++i) {
                if (tokens[i + 1] != expected_headers[i]) {
                    delete[] tokens;
                    for (int i = 0; i < max_rows; ++i) delete[] temp_data[i];
                    delete[] temp_data;
                    throw std::runtime_error("Unexpected header at column " + std::to_string(i + 1));
                }
            }
            first_line = false;
            delete[] tokens;
            continue;
        }

        if (token_count != expected_cols + 1) {
            delete[] tokens;
            for (int i = 0; i < max_rows; ++i) delete[] temp_data[i];
            delete[] temp_data;
            throw std::runtime_error("Inconsistent number of columns at line " + std::to_string(line_count + 2));
        }

        if (line_count >= max_rows) {
            int new_max_rows = max_rows * 2;
            __PROMISE__** new_temp_data = new __PROMISE__*[new_max_rows];
            for (int i = 0; i < new_max_rows; ++i) {
                new_temp_data[i] = new __PROMISE__[max_cols];
            }
            for (int i = 0; i < max_rows; ++i) {
                std::memcpy(new_temp_data[i], temp_data[i], max_cols * sizeof(__PROMISE__));
                delete[] temp_data[i];
            }
            delete[] temp_data;
            temp_data = new_temp_data;
            max_rows = new_max_rows;
        }

        for (int i = 1; i < token_count; ++i) {
            try {
                temp_data[line_count][i - 1] = std::stod(tokens[i]);
            } catch (...) {
                delete[] tokens;
                for (int i = 0; i < max_rows; ++i) delete[] temp_data[i];
                delete[] temp_data;
                throw std::runtime_error("Invalid number at line " + std::to_string(line_count + 2) + ", column " + std::to_string(i));
            }
        }
        line_count++;
        delete[] tokens;
    }

    if (line_count == 0) {
        for (int i = 0; i < max_rows; ++i) delete[] temp_data[i];
        delete[] temp_data;
        throw std::runtime_error("Empty CSV file after header");
    }

    Matrix X(line_count, expected_cols);
    for (int i = 0; i < line_count; ++i) {
        for (int j = 0; j < expected_cols; ++j) {
            X.set(i, j, temp_data[i][j]);
        }
    }

    for (int i = 0; i < max_rows; ++i) delete[] temp_data[i];
    delete[] temp_data;

    return X;
}

void preprocess_data(Matrix& X, bool standardize = true) {
    int n = X.rows, p = X.cols;
    __PROMISE__* mean = new __PROMISE__[p]();
    __PROMISE__* stddev = new __PROMISE__[p]();

    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < n; ++i) {
            mean[j] += X.get(i, j);
        }
        mean[j] /= n;
    }

    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < n; ++i) {
            X.set(i, j, X.get(i, j) - mean[j]);
        }
    }

    if (standardize) {
        for (int j = 0; j < p; ++j) {
            for (int i = 0; i < n; ++i) {
                stddev[j] += X.get(i, j) * X.get(i, j);
            }
            stddev[j] = sqrt(stddev[j] / (n - 1));
            if (stddev[j] < 1e-10) {
                std::cerr << "Warning: Near-zero variance in feature " << j << std::endl;
                stddev[j] = 1.0;
            }
        }

        for (int j = 0; j < p; ++j) {
            for (int i = 0; i < n; ++i) {
                X.set(i, j, X.get(i, j) / stddev[j]);
            }
        }
    }

    delete[] mean;
    delete[] stddev;
}

Matrix matrix_multiply(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Matrix dimensions mismatch for multiplication");
    }
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            __PROMISE__ sum = 0.0;
            for (int k = 0; k < A.cols; ++k) {
                sum += A.get(i, k) * B.get(k, j);
            }
            C.set(i, j, sum);
        }
    }
    return C;
}

Matrix compute_AAt(const Matrix& A) {
    int m = A.rows;
    Matrix AAt(m, m);
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            __PROMISE__ sum = 0.0;
            for (int k = 0; k < A.cols; ++k) {
                sum += A.get(i, k) * A.get(j, k);
            }
            AAt.set(i, j, sum);
            if (i != j) {
                AAt.set(j, i, sum);
            }
        }
    }
    return AAt;
}

void jacobi_eigendecomposition(const Matrix& A, Matrix& eigenvectors, __PROMISE__* eigenvalues, int max_iter = 500) {
    int n = A.rows;
    if (n != A.cols) {
        throw std::runtime_error("Matrix must be square for eigendecomposition");
    }
    eigenvectors.resize(n, n);
    double temp1 = 1.0, temp2 = 0.0;
    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            eigenvectors.set(i, j, (i == j) ? temp1 : temp2);
        }
    }
    Matrix B(A);

    __PROMISE__ norm = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            norm += B.get(i, j) * B.get(i, j);
        }
    }


    double temp3 = 2.0;
    norm = sqrt(norm);
    __PROMISE__ tol = 1e-12 * norm;

    for (int iter = 0; iter < max_iter; ++iter) {
        bool converged = true;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                __PROMISE__ a_pp = B.get(p, p), a_pq = B.get(p, q), a_qq = B.get(q, q);
                if (abs(a_pq) < tol) continue;
                converged = false;
                
                __PROMISE__ temp4 = -1;
                __PROMISE__ temp5 = 0;
                __PROMISE__ tau = (a_qq - a_pp) / (temp3 * a_pq);
                __PROMISE__ t = (tau >= temp5 ? temp1 : temp4) / (abs(tau) + sqrt(temp1 + tau * tau));
                __PROMISE__ c = temp1 / sqrt(1.0 + t * t);
                __PROMISE__ s = t * c;

                for (int i = 0; i < n; ++i) {
                    __PROMISE__ bp = B.get(i, p), bq = B.get(i, q);
                    B.set(i, p, c * bp - s * bq);
                    if (i != p && i != q) {
                        B.set(p, i, c * bp - s * bq);
                    }
                    B.set(i, q, s * bp + c * bq);
                    if (i != p && i != q) {
                        B.set(q, i, s * bp + c * bq);
                    }
                }
                B.set(p, p, c * c * a_pp - 2 * c * s * a_pq + s * s * a_qq);
                B.set(q, q, s * s * a_pp + 2 * c * s * a_pq + c * c * a_qq);
                B.set(p, q, 0.0);
                B.set(q, p, 0.0);

                for (int i = 0; i < n; ++i) {
                    __PROMISE__ vp = eigenvectors.get(i, p), vq = eigenvectors.get(i, q);
                    eigenvectors.set(i, p, c * vp - s * vq);
                    eigenvectors.set(i, q, s * vp + c * vq);
                }
            }
        }
        if (converged) {
            std::cout << "Jacobi converged in " << iter + 1 << " iterations" << std::endl;
            break;
        }
    }

    for (int i = 0; i < n; ++i) {
        eigenvalues[i] = B.get(i, i);
        if (eigenvalues[i] < 0.0 && abs(eigenvalues[i]) > 1e-10 * norm) {
            std::cerr << "Warning: Significant negative eigenvalue: " << eigenvalues[i] << std::endl;
        }
    }

    int* indices = new int[n];
    for (int i = 0; i < n; ++i) indices[i] = i;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (eigenvalues[indices[i]] < eigenvalues[indices[j]]) {
                std::swap(indices[i], indices[j]);
            }
        }
    }
    Matrix temp_eigenvectors(n, n);
    __PROMISE__* temp_eigenvalues = new __PROMISE__[n];
    for (int i = 0; i < n; ++i) {
        temp_eigenvalues[i] = eigenvalues[indices[i]];
        for (int j = 0; j < n; ++j) {
            temp_eigenvectors.set(j, i, eigenvectors.get(j, indices[i]));
        }
    }
    eigenvectors = temp_eigenvectors;
    
    delete[] indices;
    delete[] temp_eigenvalues;
}

struct NystromResult {
    Matrix eigenvectors;
    __PROMISE__* eigenvalues;
    Matrix approximation;
    __PROMISE__ reconstruction_error;
    int n_components;

    NystromResult(int m, int k)
        : eigenvectors(m, k), approximation(m, m), reconstruction_error(0.0), n_components(k) {
        eigenvalues = new __PROMISE__[k]();
    }

    NystromResult(const NystromResult& other)
        : eigenvectors(other.eigenvectors), approximation(other.approximation),
          reconstruction_error(other.reconstruction_error), n_components(other.n_components) {
        eigenvalues = new __PROMISE__[n_components];
        std::memcpy(eigenvalues, other.eigenvalues, n_components * sizeof(__PROMISE__));
    }

    NystromResult& operator=(const NystromResult& other) {
        if (this != &other) {
            delete[] eigenvalues;
            eigenvectors = other.eigenvectors;
            approximation = other.approximation;
            reconstruction_error = other.reconstruction_error;
            n_components = other.n_components;
            eigenvalues = new __PROMISE__[n_components];
            std::memcpy(eigenvalues, other.eigenvalues, n_components * sizeof(__PROMISE__));
        }
        return *this;
    }

    NystromResult(NystromResult&& other) noexcept
        : eigenvectors(std::move(other.eigenvectors)),
          approximation(std::move(other.approximation)),
          reconstruction_error(other.reconstruction_error),
          n_components(other.n_components),
          eigenvalues(other.eigenvalues) {
        other.eigenvalues = nullptr;
    }

    NystromResult& operator=(NystromResult&& other) noexcept {
        if (this != &other) {
            delete[] eigenvalues;
            eigenvectors = std::move(other.eigenvectors);
            approximation = std::move(other.approximation);
            reconstruction_error = other.reconstruction_error;
            n_components = other.n_components;
            eigenvalues = other.eigenvalues;
            other.eigenvalues = nullptr;
        }
        return *this;
    }

    ~NystromResult() {
        delete[] eigenvalues;
    }
};

void modified_gram_schmidt(Matrix& V, int n_components) {
    int n = V.rows;
    for (int j = 0; j < n_components; ++j) {
        __PROMISE__ norm = 0.0;
        for (int i = 0; i < n; ++i) {
            norm += V.get(i, j) * V.get(i, j);
        }
        norm = sqrt(norm);
        if (norm > 1e-10) {
            for (int i = 0; i < n; ++i) {
                V.set(i, j, V.get(i, j) / norm);
            }
        }

        for (int k = j + 1; k < n_components; ++k) {
            __PROMISE__ dot = 0.0;
            for (int i = 0; i < n; ++i) {
                dot += V.get(i, j) * V.get(i, k);
            }
            for (int i = 0; i < n; ++i) {
                V.set(i, k, V.get(i, k) - dot * V.get(i, j));
            }
        }
    }
}

NystromResult nystrom_approximation(const Matrix& A, int n_components, int sample_size) {
    int m = A.rows, n = A.cols;
    if (n_components > m || n_components <= 0 || sample_size > m || sample_size < n_components) {
        throw std::runtime_error("Invalid n_components or sample_size");
    }

    Matrix AAt = compute_AAt(A);
    double trace = 0.0;
    for (int i = 0; i < m; ++i) {
        trace += AAt.get(i, i);
    }
    __PROMISE__ reg_param = trace / m * 1e-6;

    int* indices = new int[m];
    for (int i = 0; i < m; ++i) indices[i] = i;

    std::mt19937 g(42);
    std::shuffle(indices, indices + m, g);
    int* sampled_indices = new int[sample_size];
    std::memcpy(sampled_indices, indices, sample_size * sizeof(int));

    Matrix W(sample_size, sample_size);
    for (int i = 0; i < sample_size; ++i) {
        for (int j = 0; j < sample_size; ++j) {
            W.set(i, j, AAt.get(sampled_indices[i], sampled_indices[j]));
        }
    }
    for (int i = 0; i < sample_size; ++i) {
        W.set(i, i, W.get(i, i) + reg_param);
    }

    Matrix W_eigenvectors(sample_size, sample_size);
    __PROMISE__* W_eigenvalues = new __PROMISE__[sample_size]();
    jacobi_eigendecomposition(W, W_eigenvectors, W_eigenvalues);

    Matrix C(m, sample_size);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < sample_size; ++j) {
            C.set(i, j, AAt.get(i, sampled_indices[j]));
        }
    }

    double temp8 = 1.0;
    double temp9 = 0.0;
    __PROMISE__ threshold = trace / m * 1e-8;
    Matrix W_inv_sqrt(sample_size, sample_size);
    for (int i = 0; i < sample_size; ++i) {
        __PROMISE__ sqrt_eigenval = (W_eigenvalues[i] > threshold) ? temp8 / sqrt(W_eigenvalues[i]) : temp9;
        for (int j = 0; j < sample_size; ++j) {
            W_inv_sqrt.set(i, j, W_eigenvectors.get(i, j) * sqrt_eigenval);
        }
    }

    Matrix U = matrix_multiply(C, W_inv_sqrt);

    NystromResult result(m, n_components);
    for (int i = 0; i < n_components; ++i) {
        if (W_eigenvalues[i] < threshold) {
            std::cerr << "Warning: Small eigenvalue at index " << i << ": " << W_eigenvalues[i] << std::endl;
        }
        result.eigenvalues[i] = W_eigenvalues[i] * (m / static_cast<__PROMISE__>(sample_size));
        for (int j = 0; j < m; ++j) {
            result.eigenvectors.set(j, i, U.get(j, i));
        }
    }

    modified_gram_schmidt(result.eigenvectors, n_components);

    Matrix U_k(m, n_components);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n_components; ++j) {
            U_k.set(i, j, result.eigenvectors.get(i, j));
        }
    }
    Matrix U_kT(n_components, m);
    for (int i = 0; i < n_components; ++i) {
        for (int j = 0; j < m; ++j) {
            U_kT.set(i, j, U_k.get(j, i));
        }
    }

    double temp3 = 0.0;
    Matrix Lambda_sqrt(n_components, n_components);
    for (int i = 0; i < n_components; ++i) {
        Lambda_sqrt.set(i, i, sqrt(max(temp3, result.eigenvalues[i])));
    }

    Matrix temp = matrix_multiply(U_k, Lambda_sqrt);
    result.approximation = matrix_multiply(temp, U_kT);

    __PROMISE__ norm_approx = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            norm_approx += result.approximation.get(i, j) * result.approximation.get(i, j);
        }
    }
    norm_approx = sqrt(norm_approx);
    std::cout << "Frobenius norm of approximation for n_components=" << n_components << ": " << norm_approx << std::endl;

    Matrix U_kT_U_k = matrix_multiply(U_kT, U_k);
    std::cout << "Orthonormality check (U_k^T * U_k):\n";
    __PROMISE__ temp1 = 1.0, temp2 = 0.0;
    for (int i = 0; i < n_components; ++i) {
        for (int j = 0; j < n_components; ++j) {
            __PROMISE__ expected = (i == j) ? temp1 : temp2;
            if (abs(U_kT_U_k.get(i, j) - expected) > 1e-8) {
                std::cout << "Warning: U_k not orthonormal at (" << i << "," << j << "): " << U_kT_U_k.get(i, j) << std::endl;
            }
        }
    }

    result.reconstruction_error = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            __PROMISE__ diff = AAt.get(i, j) - result.approximation.get(i, j);
            result.reconstruction_error += diff * diff;
        }
    }
    result.reconstruction_error = sqrt(result.reconstruction_error);

    __PROMISE__ norm_AAt = sqrt(trace);
    __PROMISE__ norm_AAt_approx = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            norm_AAt_approx += result.approximation.get(i, j) * result.approximation.get(i, j);
        }
    }
    norm_AAt_approx = sqrt(norm_AAt_approx);
    std::cout << "Frobenius norm of AAt: " << norm_AAt << ", AAt_approx: " << norm_AAt_approx << std::endl;

    Matrix full_AAt = compute_AAt(A);
    Matrix full_eigenvectors(m, m);
    __PROMISE__* full_eigenvalues = new __PROMISE__[m]();
    jacobi_eigendecomposition(full_AAt, full_eigenvectors, full_eigenvalues);
    __PROMISE__ total_variance = 0.0;
    __PROMISE__ temp4 = 0.0;

    for (int i = 0; i < m; ++i) {
        total_variance += max(temp4, full_eigenvalues[i]);
    }
    __PROMISE__ explained_variance = 0.0;
    
    for (int i = 0; i < n_components; ++i) {
        explained_variance += max(temp4, result.eigenvalues[i]);
    }

    delete[] indices;
    delete[] sampled_indices;
    delete[] W_eigenvalues;
    delete[] full_eigenvalues;

    return result;
}


int main(int argc, char* argv[]) {
    try {
        std::string input_file = "diabetes_features.csv";
        std::string output_file = "nystrom_results.csv";
        int n_components = 5;
        bool standardize = false;
        __PROMISE__ sample_ratio = 0.8;

        Matrix X = read_csv(input_file);

        int sample_size = max(n_components, static_cast<int>(X.rows * sample_ratio));
        if (sample_size > X.rows) {
            sample_size = X.rows;
        }

        preprocess_data(X, standardize);
        std::cout << "Data preprocessed with standardization: " << (standardize ? "true" : "false") << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        NystromResult result = nystrom_approximation(X, n_components, sample_size);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Nyström approximation completed in " << duration.count() << " ms\n";
        std::cout << "Number of components: " << n_components << "\n";
        std::cout << "Sample size: " << sample_size << "\n";
        std::cout << "Reconstruction error: " << result.reconstruction_error << "\n";
        __PROMISE__ error = result.reconstruction_error;

        PROMISE_CHECK_VAR(error);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}