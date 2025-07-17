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

class Matrix {
public:
    double* data;
    int rows, cols;

    Matrix() : rows(0), cols(0), data(nullptr) {}
    Matrix(int r, int c) : rows(r), cols(c) {
        data = new double[r * c]();
    }

    ~Matrix() {
        delete[] data;
    }

    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        data = new double[rows * cols];
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = other.data[i];
        }
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            delete[] data;
            rows = other.rows;
            cols = other.cols;
            data = new double[rows * cols];
            for (int i = 0; i < rows * cols; ++i) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }

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
        delete[] data;
        rows = r;
        cols = c;
        data = new double[r * c]();
    }
};

Matrix read_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Temporary storage for rows
    double** temp_data = nullptr;
    std::string line;
    bool first_line = true;
    const char* expected_headers[] = {"age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"};
    int expected_cols = 10;
    int line_count = 0;
    int max_rows = 1000; // Initial capacity
    int current_rows = 0;

    temp_data = new double*[max_rows];

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string* tokens = new std::string[expected_cols + 1];
        int token_count = 0;
        std::string val;
        while (std::getline(ss, val, ',')) {
            if (token_count < expected_cols + 1) {
                tokens[token_count++] = val;
            }
        }

        if (first_line) {
            if (token_count < expected_cols + 1) {
                delete[] tokens;
                for (int i = 0; i < current_rows; ++i) delete[] temp_data[i];
                delete[] temp_data;
                throw std::runtime_error("Invalid header: too few columns");
            }
            for (int i = 0; i < expected_cols; ++i) {
                if (tokens[i + 1] != expected_headers[i]) {
                    delete[] tokens;
                    for (int i = 0; i < current_rows; ++i) delete[] temp_data[i];
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
            for (int i = 0; i < current_rows; ++i) delete[] temp_data[i];
            delete[] temp_data;
            throw std::runtime_error("Inconsistent number of columns at line " + std::to_string(line_count + 2));
        }

        if (current_rows >= max_rows) {
            max_rows *= 2;
            double** new_temp_data = new double*[max_rows];
            for (int i = 0; i < current_rows; ++i) {
                new_temp_data[i] = temp_data[i];
            }
            delete[] temp_data;
            temp_data = new_temp_data;
        }

        temp_data[current_rows] = new double[expected_cols];
        for (int i = 1; i < token_count; ++i) {
            try {
                temp_data[current_rows][i - 1] = std::stod(tokens[i]);
            } catch (...) {
                delete[] tokens;
                for (int i = 0; i <= current_rows; ++i) delete[] temp_data[i];
                delete[] temp_data;
                throw std::runtime_error("Invalid number at line " + std::to_string(line_count + 2) + ", column " + std::to_string(i));
            }
        }
        current_rows++;
        line_count++;
        delete[] tokens;
    }

    if (current_rows == 0) {
        delete[] temp_data;
        throw std::runtime_error("Empty CSV file after header");
    }

    Matrix X(current_rows, expected_cols);
    for (int i = 0; i < current_rows; ++i) {
        for (int j = 0; j < expected_cols; ++j) {
            X(i, j) = temp_data[i][j];
        }
        delete[] temp_data[i];
    }
    delete[] temp_data;

    std::cout << "Loaded Diabetes data: " << current_rows << " samples, " << expected_cols << " features" << std::endl;
    return X;
}

void preprocess_data(Matrix& X, bool standardize = true) {
    int n = X.rows, p = X.cols;
    double* mean = new double[p]();
    double* stddev = new double[p]();

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

    delete[] mean;
    delete[] stddev;
}

double dot(const double* a, const double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double norm(const double* v, int n) {
    double d = dot(v, v, n);
    if (std::isnan(d) || std::isinf(d) || d < 0.0) {
        throw std::runtime_error("Invalid norm in SVD");
    }
    return std::sqrt(d);
}

void svd(const Matrix& X, Matrix& U, double* S, Matrix& Vt, int max_iter = 100) {
    int m = X.rows, n = X.cols;
    int k = std::min({m, n, U.cols});
    if (U.cols != Vt.rows || k <= 0) {
        throw std::runtime_error("SVD output dimensions mismatch");
    }

    U.resize(m, k);
    for (int i = 0; i < k; ++i) S[i] = 0.0;
    Vt.resize(k, n);

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

    Matrix A = X;

    for (int iter = 0; iter < max_iter; ++iter) {
        bool converged = true;
        for (int p = 0; p < k; ++p) {
            for (int q = p + 1; q < k; ++q) {
                double a_pp = 0.0, a_pq = 0.0, a_qq = 0.0;
                for (int i = 0; i < m; ++i) {
                    a_pp += A(i, p) * A(i, p);
                    a_pq += A(i, p) * A(i, q);
                    a_qq += A(i, q) * A(i, q);
                }

                if (std::abs(a_pq) < 1e-14 * std::max(a_pp, a_qq)) continue;
                converged = false;

                double tau = (a_qq - a_pp) / (2.0 * a_pq);
                double t = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                for (int i = 0; i < m; ++i) {
                    double ap = A(i, p), aq = A(i, q);
                    A(i, p) = c * ap - s * aq;
                    A(i, q) = s * ap + c * aq;
                }

                for (int i = 0; i < n; ++i) {
                    double vp = Vt(p, i), vq = Vt(q, i);
                    Vt(p, i) = c * vp - s * vq;
                    Vt(q, i) = s * vp + c * vq;
                }

                for (int i = 0; i < m; ++i) {
                    double up = U(i, p), uq = U(i, q);
                    U(i, p) = c * up - s * uq;
                    U(i, q) = s * up + c * uq;
                }
            }
        }
        if (converged) break;
    }

    double* col = new double[m];
    for (int i = 0; i < k; ++i) {
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
    delete[] col;

    int* indices = new int[k];
    for (int i = 0; i < k; ++i) indices[i] = i;
    for (int i = 0; i < k - 1; ++i) {
        for (int j = i + 1; j < k; ++j) {
            if (S[indices[i]] < S[indices[j]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }

    Matrix U_temp(m, k), Vt_temp(k, n);
    double* S_temp = new double[k];
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
    for (int i = 0; i < k; ++i) S[i] = S_temp[i];
    Vt = Vt_temp;
    delete[] S_temp;
    delete[] indices;
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

double compute_reconstruction_error(const Matrix& X, const Matrix& U, const double* S, const Matrix& Vt) {
    int m = X.rows, n = X.cols, k = U.cols;

    Matrix S_diag(k, k);
    for (int i = 0; i < k; ++i) {
        S_diag(i, i) = S[i];
    }
    Matrix temp = matrix_multiply(U, S_diag);
    Matrix X_recon = matrix_multiply(temp, Vt);

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
    double* explained_variance;
    Matrix scores;
    double reconstruction_error;
    int n_components;

    PCAResult(int comp_rows, int comp_cols, int score_rows, int score_cols)
        : components(comp_rows, comp_cols), scores(score_rows, score_cols), reconstruction_error(0.0), n_components(comp_cols) {
        explained_variance = new double[n_components]();
    }

    ~PCAResult() {
        delete[] explained_variance;
    }
};

PCAResult perform_pca(const Matrix& X, int n_components = -1) {
    int m = X.rows, n = X.cols;
    if (n_components == -1) n_components = std::min(m, n);
    if (n_components > std::min(m, n) || n_components <= 0) {
        throw std::runtime_error("Invalid number of components");
    }

    Matrix U(m, n_components);
    double* S = new double[n_components]();
    Matrix Vt(n_components, n);
    svd(X, U, S, Vt);

    PCAResult result(n_components, n, m, n_components);
    result.components = Vt;

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

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n_components; ++j) {
            result.scores(i, j) = U(i, j) * S[j];
        }
    }

    result.reconstruction_error = compute_reconstruction_error(X, U, S, Vt);

    delete[] S;
    return result;
}

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
    for (int i = 0; i < result.n_components; ++i) {
        file << result.explained_variance[i];
        if (i < result.n_components - 1) file << ",";
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
        int n_components = 4;
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
        for (int i = 0; i < result.n_components; ++i) {
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