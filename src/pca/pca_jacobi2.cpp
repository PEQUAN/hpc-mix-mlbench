#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <string>
#include <chrono>

class Matrix {
public:
    std::vector<double> data;
    int rows, cols;

    Matrix() : rows(0), cols(0), data() {}
    Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

    double& operator()(int i, int j) {
        if (i >= rows || j >= cols || i < 0 || j < 0) {
            throw std::runtime_error("Matrix index out of bounds");
        }
        return data[i * cols + j];
    }
    const double& operator()(int i, int j) const {
        if (i >= rows || j >= cols || i < 0 || j < 0) {
            throw std::runtime_error("Matrix index out of bounds");
        }
        return data[i * cols + j];
    }

    void resize(int r, int c) {
        rows = r;
        cols = c;
        data.resize(r * c, 0.0);
    }
};

Matrix read_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::vector<double>> temp_data;
    std::string line;
    bool first_line = true;
    std::vector<std::string> expected_headers = {"age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"};
    int expected_cols = expected_headers.size();
    int line_count = 0;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> tokens;
        std::string val;
        while (std::getline(ss, val, ',')) {
            tokens.push_back(val);
        }

        if (first_line) {
            if (tokens.size() < expected_cols + 1) {
                throw std::runtime_error("Invalid header: too few columns");
            }
            for (size_t i = 0; i < expected_cols; ++i) {
                if (tokens[i + 1] != expected_headers[i]) {
                    throw std::runtime_error("Unexpected header at column " + std::to_string(i + 1) + ": got " + tokens[i + 1] + ", expected " + expected_headers[i]);
                }
            }
            first_line = false;
            continue;
        }

        if (tokens.size() != expected_cols + 1) {
            throw std::runtime_error("Inconsistent number of columns at line " + std::to_string(line_count + 2));
        }

        std::vector<double> row;
        for (size_t i = 1; i < tokens.size(); ++i) {
            try {
                row.push_back(std::stod(tokens[i]));
            } catch (...) {
                throw std::runtime_error("Invalid number at line " + std::to_string(line_count + 2) + ", column " + std::to_string(i));
            }
        }
        temp_data.push_back(row);
        line_count++;
    }

    int rows = temp_data.size();
    if (rows == 0) {
        throw std::runtime_error("Empty CSV file after header");
    }

    Matrix X(rows, expected_cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < expected_cols; ++j) {
            X(i, j) = temp_data[i][j];
        }
    }

    std::cout << "Loaded Diabetes data: " << rows << " samples, " << expected_cols << " features" << std::endl;
    return X;
}

void preprocess_data(Matrix& X, bool standardize = true) {
    int n = X.rows, p = X.cols;
    std::vector<double> mean(p, 0.0);
    std::vector<double> stddev(p, 0.0);

    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < n; ++i) {
            mean[j] += X(i, j);
        }
        mean[j] /= n;
    }

    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < n; ++i) {
            X(i, j) -= mean[j];
        }
    }

    if (standardize) {
        for (int j = 0; j < p; ++j) {
            for (int i = 0; i < n; ++i) {
                stddev[j] += X(i, j) * X(i, j);
            }
            stddev[j] = std::sqrt(stddev[j] / (n - 1));
            if (stddev[j] < 1e-10) {
                std::cerr << "Warning: Near-zero variance in feature " << j << std::endl;
                stddev[j] = 1.0;
            }
        }

        for (int j = 0; j < p; ++j) {
            for (int i = 0; i < n; ++i) {
                X(i, j) /= stddev[j];
            }
        }
    }
}

double dot(const std::vector<double>& a, const std::vector<double>& b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double norm(const std::vector<double>& v, int n) {
    double d = dot(v, v, n);
    if (std::isnan(d) || std::isinf(d) || d < 0.0) {
        throw std::runtime_error("Invalid norm in SVD");
    }
    return std::sqrt(d);
}

// SVD to respect n_components
void svd(const Matrix& X, Matrix& U, std::vector<double>& S, Matrix& Vt, int max_iter = 100) {
    int m = X.rows, n = X.cols;
    int k = std::min({m, n, U.cols}); // Use requested number of components
    if (U.cols != Vt.rows || S.size() != static_cast<size_t>(k)) {
        throw std::runtime_error("SVD output dimensions mismatch");
    }

    // Resize outputs if necessary
    U.resize(m, k);
    S.assign(k, 0.0);
    Vt.resize(k, n);

    // Initialize U and Vt
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            U(i, j) = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            Vt(i, j) = (i == j && j < k) ? 1.0 : 0.0;
        }
    }

    // Copy X to A
    Matrix A = X;

    // Jacobi SVD for k components
    for (int iter = 0; iter < max_iter; ++iter) {
        bool converged = true;
        for (int p = 0; p < k; ++p) {
            for (int q = p + 1; q < k; ++q) {
                // Compute 2x2 submatrix for A^T A
                double a_pp = 0.0, a_pq = 0.0, a_qq = 0.0;
                for (int i = 0; i < m; ++i) {
                    a_pp += A(i, p) * A(i, p);
                    a_pq += A(i, p) * A(i, q);
                    a_qq += A(i, q) * A(i, q);
                }

                if (std::abs(a_pq) < 1e-14 * std::max(a_pp, a_qq)) continue;
                converged = false;

                // Compute Jacobi rotation
                double tau = (a_qq - a_pp) / (2.0 * a_pq);
                double t = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                // Update A
                for (int i = 0; i < m; ++i) {
                    double ap = A(i, p), aq = A(i, q);
                    A(i, p) = c * ap - s * aq;
                    A(i, q) = s * ap + c * aq;
                }

                // Update Vt
                for (int i = 0; i < n; ++i) {
                    double vp = Vt(p, i), vq = Vt(q, i);
                    Vt(p, i) = c * vp - s * vq;
                    Vt(q, i) = s * vp + c * vq;
                }

                // Update U
                for (int i = 0; i < m; ++i) {
                    double up = U(i, p), uq = U(i, q);
                    U(i, p) = c * up - s * uq;
                    U(i, q) = s * up + c * uq;
                }
            }
        }
        if (converged) break;
    }

    // Extract singular values
    for (int i = 0; i < k; ++i) {
        std::vector<double> col(m);
        for (int j = 0; j < m; ++j) col[j] = A(j, i);
        S[i] = norm(col, m);
        if (S[i] > 1e-10) {
            for (int j = 0; j < m; ++j) {
                U(j, i) = A(j, i) / S[i];
            }
        } else {
            S[i] = 0.0;
        }
    }

    // Sort singular values and vectors
    std::vector<int> indices(k);
    for (int i = 0; i < k; ++i) indices[i] = i;
    std::sort(indices.begin(), indices.end(),
              [&S](int a, int b) { return S[a] > S[b]; });

    Matrix U_temp(m, k), Vt_temp(k, n);
    std::vector<double> S_temp(k);
    for (int i = 0; i < k; ++i) {
        S_temp[i] = S[indices[i]];
        for (int j = 0; j < m; ++j) {
            U_temp(j, i) = U(j, indices[i]);
        }
        for (int j = 0; j < n; ++j) {
            Vt_temp(i, j) = Vt(indices[i], j);
        }
    }
    U = U_temp;
    S = S_temp;
    Vt = Vt_temp;
}

Matrix matrix_multiply(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Matrix dimensions mismatch for multiplication");
    }
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            for (int k = 0; k < A.cols; ++k) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return C;
}

// Modified compute_reconstruction_error
double compute_reconstruction_error(const Matrix& X, const Matrix& U, const std::vector<double>& S, const Matrix& Vt) {
    int m = X.rows, n = X.cols, k = S.size();

    // Compute X_recon = U * S * Vt
    Matrix S_diag(k, k);
    for (int i = 0; i < k; ++i) {
        S_diag(i, i) = S[i];
    }
    Matrix temp = matrix_multiply(U, S_diag); // U * S
    Matrix X_recon = matrix_multiply(temp, Vt); // (U * S) * Vt

    // Compute Frobenius norm of X - X_recon
    double error = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double diff = X(i, j) - X_recon(i, j);
            error += diff * diff;
        }
    }
    return error;
}

struct PCAResult {
    Matrix components;
    std::vector<double> explained_variance;
    Matrix scores;
    double reconstruction_error;

    PCAResult(int comp_rows, int comp_cols, int score_rows, int score_cols)
        : components(comp_rows, comp_cols), explained_variance(), scores(score_rows, score_cols), reconstruction_error(0.0) {}
};

PCAResult perform_pca(const Matrix& X, int n_components = -1) {
    int m = X.rows, n = X.cols;
    if (n_components == -1) n_components = std::min(m, n);
    if (n_components > std::min(m, n) || n_components <= 0) {
        throw std::runtime_error("Invalid number of components");
    }

    // Compute SVD
    Matrix U(m, n_components);
    std::vector<double> S(n_components);
    Matrix Vt(n_components, n);
    svd(X, U, S, Vt);

    // Initialize PCAResult
    PCAResult result(n_components, n, m, n_components);
    result.components = Vt;

    // Explained variance
    result.explained_variance.resize(n_components);
    double total_variance = 0.0;
    for (int i = 0; i < n_components; ++i) {
        result.explained_variance[i] = (S[i] * S[i]) / (m - 1);
        total_variance += result.explained_variance[i];
    }
    if (total_variance > 0) {
        for (int i = 0; i < n_components; ++i) {
            result.explained_variance[i] /= total_variance;
        }
    }

    // Compute scores: U * S
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n_components; ++j) {
            result.scores(i, j) = U(i, j) * S[j];
        }
    }

    // Compute reconstruction error
    result.reconstruction_error = compute_reconstruction_error(X, U, S, Vt);

    return result;
}

// write_pca_results (unchanged)
void write_pca_results(const PCAResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + filename);
    }

    file << std::fixed << std::setprecision(6);
    file << "Principal Components\n";
    for (int i = 0; i < result.components.rows; ++i) {
        for (int j = 0; j < result.components.cols; ++j) {
            file << result.components(i, j);
            if (j < result.components.cols - 1) file << ",";
        }
        file << "\n";
    }

    file << "\nExplained Variance Ratio\n";
    for (size_t i = 0; i < result.explained_variance.size(); ++i) {
        file << result.explained_variance[i];
        if (i < result.explained_variance.size() - 1) file << ",";
    }
    file << "\n";

    file << "\nScores\n";
    for (int i = 0; i < result.scores.rows; ++i) {
        for (int j = 0; j < result.scores.cols; ++j) {
            file << result.scores(i, j);
            if (j < result.scores.cols - 1) file << ",";
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
        std::string output_file = (argc > 2) ? argv[2] : "pca_results.csv";
        int n_components = 6;
        bool standardize = false;

        Matrix X = read_csv(input_file);
        if (n_components == -1) {
            n_components = std::min(X.rows, X.cols);
        }
        if (n_components > X.cols || n_components <= 0) {
            throw std::runtime_error("Invalid n_components: " + std::to_string(n_components));
        }
        preprocess_data(X, standardize);

        auto start = std::chrono::high_resolution_clock::now();
        PCAResult result = perform_pca(X, n_components);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "PCA completed in " << duration.count() << " ms\n";
        std::cout << "Explained variance ratio:\n";
        for (size_t i = 0; i < result.explained_variance.size(); ++i) {
            std::cout << "PC" << (i + 1) << ": " << result.explained_variance[i] * 100 << "%\n";
        }
        std::cout << "Reconstruction error: " << result.reconstruction_error << std::endl;

        write_pca_results(result, output_file);
        std::cout << "Results written to " << output_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}