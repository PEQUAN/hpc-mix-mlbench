#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <string>
#include <chrono>
#include <random>
#include <vector>

class Matrix {
public:
    std::vector<double> data;
    int rows, cols;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            data = other.data;
        }
        return *this;
    }

    Matrix(Matrix&& other) noexcept : rows(other.rows), cols(other.cols), data(std::move(other.data)) {
        other.rows = 0;
        other.cols = 0;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            rows = other.rows;
            cols = other.cols;
            data = std::move(other.data);
            other.rows = 0;
            other.cols = 0;
        }
        return *this;
    }

    double get(int i, int j) const {
        if (i >= rows || j >= cols || i < 0 || j < 0) {
            throw std::runtime_error("Matrix index out of bounds");
        }
        return data[i * cols + j];
    }

    void set(int i, int j, double value) {
        if (i >= rows || j >= cols || i < 0 || j < 0) {
            throw std::runtime_error("Matrix index out of bounds");
        }
        data[i * cols + j] = value;
    }

    void resize(int r, int c) {
        rows = r;
        cols = c;
        data.assign(r * c, 0.0);
    }
};

Matrix read_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::vector<double>> temp_data;
    int max_cols = 11;
    int line_count = 0;
    std::string line;
    bool first_line = true;
    const char* expected_headers[] = {"age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"};
    int expected_cols = 10;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> tokens;
        std::string val;
        while (std::getline(ss, val, ',')) {
            tokens.push_back(val);
        }
        int token_count = tokens.size();

        if (first_line) {
            if (token_count < expected_cols + 1) {
                throw std::runtime_error("Invalid header: too few columns");
            }
            for (int i = 0; i < expected_cols; ++i) {
                if (tokens[i + 1] != expected_headers[i]) {
                    throw std::runtime_error("Unexpected header at column " + std::to_string(i + 1));
                }
            }
            first_line = false;
            continue;
        }

        if (token_count != expected_cols + 1) {
            throw std::runtime_error("Inconsistent number of columns at line " + std::to_string(line_count + 2));
        }

        std::vector<double> row(expected_cols);
        for (int i = 1; i < token_count; ++i) {
            try {
                row[i - 1] = std::stod(tokens[i]);
            } catch (...) {
                throw std::runtime_error("Invalid number at line " + std::to_string(line_count + 2) + ", column " + std::to_string(i));
            }
        }
        temp_data.push_back(row);
        line_count++;
    }

    if (line_count == 0) {
        throw std::runtime_error("Empty CSV file after header");
    }

    Matrix X(line_count, expected_cols);
    for (int i = 0; i < line_count; ++i) {
        for (int j = 0; j < expected_cols; ++j) {
            X.set(i, j, temp_data[i][j]);
        }
    }

    std::cout << "Loaded Diabetes data: " << line_count << " samples, " << expected_cols << " features" << std::endl;
    return X;
}

void preprocess_data(Matrix& X, bool standardize = true) {
    int n = X.rows, p = X.cols;
    std::vector<double> mean(p, 0.0);
    std::vector<double> stddev(p, 0.0);

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
            stddev[j] = std::sqrt(stddev[j] / (n - 1));
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
}

Matrix matrix_multiply(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Matrix dimensions mismatch for multiplication");
    }
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < A.cols; ++k) {
                sum += A.get(i, k) * B.get(k, j);
            }
            C.set(i, j, sum);
        }
    }
    return C;
}

Matrix compute_AAt(const Matrix& A) {
// Compute AA^T
    int m = A.rows;
    Matrix AAt(m, m);
    for (int i = 0; i < m; ++i) {
        for (int j = i; j < m; ++j) {
            double sum = 0.0;
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


void jacobi_eigendecomposition(const Matrix& A, Matrix& eigenvectors, std::vector<double>& eigenvalues, int max_iter = 500) {
    // Jacobi eigendecomposition
    int n = A.rows;
    if (n != A.cols) {
        throw std::runtime_error("Matrix must be square for eigendecomposition");
    }
    eigenvectors.resize(n, n);
    eigenvalues.resize(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            eigenvectors.set(i, j, (i == j) ? 1.0 : 0.0);
        }
    }
    Matrix B(A);

    double norm = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            norm += B.get(i, j) * B.get(i, j);
        }
    }
    norm = std::sqrt(norm);
    double tol = 1e-12 * norm;

    for (int iter = 0; iter < max_iter; ++iter) {
        bool converged = true;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double a_pp = B.get(p, p), a_pq = B.get(p, q), a_qq = B.get(q, q);
                if (std::abs(a_pq) < tol) continue;
                converged = false;

                double tau = (a_qq - a_pp) / (2.0 * a_pq);
                double t = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                for (int i = 0; i < n; ++i) {
                    double bp = B.get(i, p), bq = B.get(i, q);
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
                    double vp = eigenvectors.get(i, p), vq = eigenvectors.get(i, q);
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
        if (eigenvalues[i] < 0.0 && std::abs(eigenvalues[i]) > 1e-10 * norm) {
            std::cerr << "Warning: Significant negative eigenvalue: " << eigenvalues[i] << std::endl;
        }
    }

    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (eigenvalues[indices[i]] < eigenvalues[indices[j]]) {
                std::swap(indices[i], indices[j]);
            }
        }
    }
    Matrix temp_eigenvectors(n, n);
    std::vector<double> temp_eigenvalues(n);
    for (int i = 0; i < n; ++i) {
        temp_eigenvalues[i] = eigenvalues[indices[i]];
        for (int j = 0; j < n; ++j) {
            temp_eigenvectors.set(j, i, eigenvectors.get(j, indices[i]));
        }
    }
    eigenvectors = temp_eigenvectors;
    eigenvalues = temp_eigenvalues;

    // std::cout << "Eigenvalues of W: ";
    // for (double val : eigenvalues) {
    //    std::cout << val << " ";
    // }
    // std::cout << std::endl;
}

struct NystromResult {
    Matrix eigenvectors;
    std::vector<double> eigenvalues;
    Matrix approximation;
    double reconstruction_error;
    int n_components;

    NystromResult(int m, int k)
        : eigenvectors(m, k), approximation(m, m), reconstruction_error(0.0), n_components(k) {
        eigenvalues.resize(k, 0.0);
    }

    NystromResult(const NystromResult& other)
        : eigenvectors(other.eigenvectors), eigenvalues(other.eigenvalues),
          approximation(other.approximation), reconstruction_error(other.reconstruction_error),
          n_components(other.n_components) {}

    NystromResult& operator=(const NystromResult& other) {
        if (this != &other) {
            eigenvectors = other.eigenvectors;
            eigenvalues = other.eigenvalues;
            approximation = other.approximation;
            reconstruction_error = other.reconstruction_error;
            n_components = other.n_components;
        }
        return *this;
    }

    NystromResult(NystromResult&& other) noexcept
        : eigenvectors(std::move(other.eigenvectors)),
          eigenvalues(std::move(other.eigenvalues)),
          approximation(std::move(other.approximation)),
          reconstruction_error(other.reconstruction_error),
          n_components(other.n_components) {}

    NystromResult& operator=(NystromResult&& other) noexcept {
        if (this != &other) {
            eigenvectors = std::move(other.eigenvectors);
            eigenvalues = std::move(other.eigenvalues);
            approximation = std::move(other.approximation);
            reconstruction_error = other.reconstruction_error;
            n_components = other.n_components;
        }
        return *this;
    }
};


void modified_gram_schmidt(Matrix& V, int n_components) {// Modified Gram-Schmidt
    int n = V.rows;
    for (int j = 0; j < n_components; ++j) {
        double norm = 0.0;
        for (int i = 0; i < n; ++i) {
            norm += V.get(i, j) * V.get(i, j);
        }
        norm = std::sqrt(norm);
        if (norm > 1e-10) {
            for (int i = 0; i < n; ++i) {
                V.set(i, j, V.get(i, j) / norm);
            }
        }

        for (int k = j + 1; k < n_components; ++k) {
            double dot = 0.0;
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
    // Nyström approximation
    int m = A.rows, n = A.cols;
    if (n_components > m || n_components <= 0 || sample_size > m || sample_size < n_components) {
        throw std::runtime_error("Invalid n_components or sample_size");
    }

    Matrix AAt = compute_AAt(A);
    double trace = 0.0;
    for (int i = 0; i < m; ++i) {
        trace += AAt.get(i, i);
    }
    double reg_param = trace / m * 1e-6;

    std::vector<int> indices(m);
    for (int i = 0; i < m; ++i) indices[i] = i;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    std::vector<int> sampled_indices(indices.begin(), indices.begin() + sample_size);

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
    std::vector<double> W_eigenvalues(sample_size);
    jacobi_eigendecomposition(W, W_eigenvectors, W_eigenvalues);

    // Debug: Print W eigenvalues
    // std::cout << "W_eigenvalues before scaling: ";
    // for (double val : W_eigenvalues) {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;

    Matrix C(m, sample_size);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < sample_size; ++j) {
            C.set(i, j, AAt.get(i, sampled_indices[j]));
        }
    }

    double threshold = trace / m * 1e-8; // More robust threshold
    Matrix W_inv_sqrt(sample_size, sample_size);
    for (int i = 0; i < sample_size; ++i) {
        double sqrt_eigenval = (W_eigenvalues[i] > threshold) ? 1.0 / std::sqrt(W_eigenvalues[i]) : 0.0;
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
        result.eigenvalues[i] = W_eigenvalues[i] * (m / static_cast<double>(sample_size));
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

    Matrix Lambda_sqrt(n_components, n_components);
    for (int i = 0; i < n_components; ++i) {
        Lambda_sqrt.set(i, i, std::sqrt(std::max(0.0, result.eigenvalues[i])));
    }
    
    Matrix temp = matrix_multiply(U_k, Lambda_sqrt);
    result.approximation = matrix_multiply(temp, U_kT);

    // Debug: Print norm of approximation
    double norm_approx = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            norm_approx += result.approximation.get(i, j) * result.approximation.get(i, j);
        }
    }
    norm_approx = std::sqrt(norm_approx);
    std::cout << "Frobenius norm of approximation for n_components=" << n_components << ": " << norm_approx << std::endl;

    Matrix U_kT_U_k = matrix_multiply(U_kT, U_k);
    std::cout << "Orthonormality check (U_k^T * U_k):\n";
    for (int i = 0; i < n_components; ++i) {
        for (int j = 0; j < n_components; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(U_kT_U_k.get(i, j) - expected) > 1e-8) {
                std::cout << "Warning: U_k not orthonormal at (" << i << "," << j << "): " << U_kT_U_k.get(i, j) << std::endl;
            }
        }
    }

    result.reconstruction_error = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            double diff = AAt.get(i, j) - result.approximation.get(i, j);
            result.reconstruction_error += diff * diff;
        }
    }
    result.reconstruction_error = std::sqrt(result.reconstruction_error);

    double norm_AAt = std::sqrt(trace);
    double norm_AAt_approx = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            norm_AAt_approx += result.approximation.get(i, j) * result.approximation.get(i, j);
        }
    }
    norm_AAt_approx = std::sqrt(norm_AAt_approx);
    std::cout << "Frobenius norm of AAt: " << norm_AAt << ", AAt_approx: " << norm_AAt_approx << std::endl;

    Matrix full_AAt = compute_AAt(A);
    Matrix full_eigenvectors(m, m);
    std::vector<double> full_eigenvalues(m);
    jacobi_eigendecomposition(full_AAt, full_eigenvectors, full_eigenvalues);
    double total_variance = 0.0;
    for (double val : full_eigenvalues) {
        total_variance += std::max(0.0, val);
    }
    double explained_variance = 0.0;
    for (int i = 0; i < n_components; ++i) {
        explained_variance += std::max(0.0, result.eigenvalues[i]);
    }
    std::cout << "Explained variance ratio: " << (total_variance > 0 ? explained_variance / total_variance : 0.0) << std::endl;

    return result;
}

void write_nystrom_results(const NystromResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + filename);
    }

    file << std::fixed << std::setprecision(6);
    file << "Eigenvectors\n";
    for (int i = 0; i < result.eigenvectors.rows; ++i) {
        for (int j = 0; j < result.eigenvectors.cols; ++j) {
            file << result.eigenvectors.get(i, j);
            if (j < result.eigenvectors.cols - 1) file << ",";
        }
        file << "\n";
    }

    file << "\nEigenvalues\n";
    for (size_t i = 0; i < result.eigenvalues.size(); ++i) {
        file << result.eigenvalues[i];
        if (i < result.eigenvalues.size() - 1) file << ",";
    }
    file << "\n";

    file << "\nApproximation\n";
    for (int i = 0; i < result.approximation.rows; ++i) {
        for (int j = 0; j < result.approximation.cols; ++j) {
            file << result.approximation.get(i, j);
            if (j < result.approximation.cols - 1) file << ",";
        }
        file << "\n";
    }

    file << "\nReconstruction Error\n";
    file << result.reconstruction_error << "\n";

    file.close();
}

int main(int argc, char* argv[]) {
    try {
        std::string input_file = (argc > 1) ? argv[1] : "../data/regression/diabetes_features.csv";
        std::string output_file = (argc > 2) ? argv[2] : "nystrom_results.csv";
        int n_components = 10;
        bool standardize = false; // Enable standardization for better scaling
        double sample_ratio = 0.7;

        Matrix X = read_csv(input_file);
        if (n_components > X.rows || n_components <= 0) {
            throw std::runtime_error("Invalid n_components: " + std::to_string(n_components));
        }
        int sample_size = std::max(n_components, static_cast<int>(X.rows * sample_ratio));
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

        write_nystrom_results(result, output_file);
        std::cout << "Results written to " << output_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}