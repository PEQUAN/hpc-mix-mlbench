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

    // Copy constructor
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        data = new double[rows * cols];
        for (int i = 0; i < rows * cols; ++i) {
            data[i] = other.data[i];
        }
    }

    // Assignment
    void assign(const Matrix& other) {
        if (this != &other) {
            delete[] data;
            rows = other.rows;
            cols = other.cols;
            data = new double[rows * cols];
            for (int i = 0; i < rows * cols; ++i) {
                data[i] = other.data[i];
            }
        }
    }

    // Direct indexing methods
    float get(int i, int j) const {
        if (i >= rows || j >= cols || i < 0 || j < 0) {
            throw std::runtime_error("Matrix index out of bounds");
        }
        return data[i * cols + j];
    }

    void set(int i, int j, float value) {
        if (i >= rows || j >= cols || i < 0 || j < 0) {
            throw std::runtime_error("Matrix index out of bounds");
        }
        data[i * cols + j] = value;
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

    int max_rows = 1000;
    int max_cols = 11;
    double** temp_data = new double*[max_rows];
    for (int i = 0; i < max_rows; ++i) {
        temp_data[i] = new double[max_cols];
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
        while (std::getline(ss, val, ',') && token_count < max_cols) {
            tokens[token_count++] = val;
        }

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
            delete[] tokens;
            continue;
        }

        if (token_count != expected_cols + 1) {
            throw std::runtime_error("Inconsistent number of columns at line " + std::to_string(line_count + 2));
        }

        for (int i = 1; i < token_count; ++i) {
            try {
                temp_data[line_count][i - 1] = std::stod(tokens[i]);
            } catch (...) {
                throw std::runtime_error("Invalid number at line " + std::to_string(line_count + 2) + ", column " + std::to_string(i));
            }
        }
        line_count++;
        delete[] tokens;

        if (line_count >= max_rows) {
            int new_max_rows = max_rows * 2;
            double** new_temp_data = new double*[new_max_rows];
            for (int i = 0; i < new_max_rows; ++i) {
                new_temp_data[i] = new double[max_cols];
                if (i < line_count) {
                    for (int j = 0; j < max_cols; ++j) {
                        new_temp_data[i][j] = temp_data[i][j];
                    }
                }
            }
            for (int i = 0; i < max_rows; ++i) {
                delete[] temp_data[i];
            }
            delete[] temp_data;
            temp_data = new_temp_data;
            max_rows = new_max_rows;
        }
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

    for (int i = 0; i < max_rows; ++i) {
        delete[] temp_data[i];
    }
    delete[] temp_data;

    std::cout << "Loaded Diabetes data: " << line_count << " samples, " << expected_cols << " features" << std::endl;
    return X;
}

void preprocess_data(Matrix& X, bool standardize = true) {
    int n = X.rows, p = X.cols;
    double* mean = new double[p]();
    half_float::half* stddev = new half_float::half[p]();

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

float dot(const double* a, const double* b, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float norm(const double* v, int n) {
    float d = dot(v, v, n);
    if (isnan(d) || isinf(d) || d < 0.0) {
        throw std::runtime_error("Invalid norm in SVD");
    }
    return sqrt(d);
}

void svd(const Matrix& X, Matrix& U, double* S, Matrix& Vt, int max_iter = 100) {
    int m = X.rows, n = X.cols;
    int k = min({m, n, U.cols});
    if (U.cols != Vt.rows || k <= 0) {
        throw std::runtime_error("SVD output dimensions mismatch");
    }

    U.resize(m, k);
    for (int i = 0; i < k; ++i) S[i] = 0.0;
    Vt.resize(k, n);

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            U.set(i, j, (i == j) ? 1.0 : 0.0);
        }
    }
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < n; ++j) {
            Vt.set(i, j, (i == j && j < k) ? 1.0 : 0.0);
        }
    }

    Matrix A(X);

    for (int iter = 0; iter < max_iter; ++iter) {
        bool converged = true;
        for (int p = 0; p < k; ++p) {
            for (int q = p + 1; q < k; ++q) {
                float a_pp = 0.0, a_pq = 0.0, a_qq = 0.0;
                for (int i = 0; i < m; ++i) {
                    a_pp += A.get(i, p) * A.get(i, p);
                    a_pq += A.get(i, p) * A.get(i, q);
                    a_qq += A.get(i, q) * A.get(i, q);
                }

                if (abs(a_pq) < 1e-14 * max(a_pp, a_qq)) continue;
                converged = false;

                float tau = (a_qq - a_pp) / (2.0 * a_pq);
                float t = (tau >= 0 ? 1.0 : -1.0) / (abs(tau) + sqrt(1.0 + tau * tau));
                float c = 1.0 / sqrt(1.0 + t * t);
                float s = t * c;

                for (int i = 0; i < m; ++i) {
                    float ap = A.get(i, p), aq = A.get(i, q);
                    A.set(i, p, c * ap - s * aq);
                    A.set(i, q, s * ap + c * aq);
                }

                for (int i = 0; i < n; ++i) {
                    float vp = Vt.get(p, i), vq = Vt.get(q, i);
                    Vt.set(p, i, c * vp - s * vq);
                    Vt.set(q, i, s * vp + c * vq);
                }

                for (int i = 0; i < m; ++i) {
                    float up = U.get(i, p), uq = U.get(i, q);
                    U.set(i, p, c * up - s * uq);
                    U.set(i, q, s * up + c * uq);
                }
            }
        }
        if (converged) break;
    }

    double* col = new double[m];
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < m; ++j) col[j] = A.get(j, i);
        S[i] = norm(col, m);
        if (S[i] > 1e-10) {
            for (int j = 0; j < m; ++j) {
                U.set(j, i, A.get(j, i) / S[i]);
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
                std::swap(indices[i], indices[j]);
            }
        }
    }

    Matrix U_temp(m, k), Vt_temp(k, n);
    flx::floatx<4, 3>* S_temp = new flx::floatx<4, 3>[k];
    for (int i = 0; i < k; ++i) {
        S_temp[i] = S[indices[i]];
        for (int j = 0; j < m; ++j) {
            U_temp.set(j, i, U.get(j, indices[i]));
        }
        for (int j = 0; j < n; ++j) {
            Vt_temp.set(i, j, Vt.get(indices[i], j));
        }
    }
    U.assign(U_temp);
    for (int i = 0; i < k; ++i) S[i] = S_temp[i];
    Vt.assign(Vt_temp);

    delete[] indices;
    delete[] S_temp;
}

Matrix matrix_multiply(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Matrix dimensions mismatch for multiplication");
    }
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            for (int k = 0; k < A.cols; ++k) {
                C.set(i, j, C.get(i, j) + A.get(i, k) * B.get(k, j));
            }
        }
    }
    return C;
}

float compute_reconstruction_error(const Matrix& X, const Matrix& U, const double* S, const Matrix& Vt) {
    int m = X.rows, n = X.cols, k = U.cols;

    Matrix S_diag(k, k);
    for (int i = 0; i < k; ++i) {
        S_diag.set(i, i, S[i]);
    }
    Matrix temp = matrix_multiply(U, S_diag);
    Matrix X_recon = matrix_multiply(temp, Vt);

    double error = 0.0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float diff = X.get(i, j) - X_recon.get(i, j);
            error += diff * diff;
        }
    }
    return error;
}

struct PCAResult {
    Matrix components;
    double* explained_variance;
    Matrix scores;
    flx::floatx<4, 3> reconstruction_error;
    int n_components;

    PCAResult(int comp_rows, int comp_cols, int score_rows, int score_cols)
        : components(comp_rows, comp_cols), scores(score_rows, score_cols), reconstruction_error(0.0), n_components(comp_rows) {
        explained_variance = new double[comp_rows]();
    }

    ~PCAResult() {
        delete[] explained_variance;
    }

    PCAResult(const PCAResult& other)
        : components(other.components), scores(other.scores), reconstruction_error(other.reconstruction_error), n_components(other.n_components) {
        explained_variance = new double[n_components];
        for (int i = 0; i < n_components; ++i) {
            explained_variance[i] = other.explained_variance[i];
        }
    }

    void assign(const PCAResult& other) {
        if (this != &other) {
            components.assign(other.components);
            scores.assign(other.scores);
            reconstruction_error = other.reconstruction_error;
            n_components = other.n_components;
            delete[] explained_variance;
            explained_variance = new double[n_components];
            for (int i = 0; i < n_components; ++i) {
                explained_variance[i] = other.explained_variance[i];
            }
        }
    }
};

PCAResult perform_pca(const Matrix& X, int n_components = -1) {
    int m = X.rows, n = X.cols;
    if (n_components == -1) n_components = min(m, n);
    if (n_components > min(m, n) || n_components <= 0) {
        throw std::runtime_error("Invalid number of components");
    }

    Matrix U(m, n_components);
    double* S = new double[n_components]();
    Matrix Vt(n_components, n);
    svd(X, U, S, Vt);

    PCAResult result(n_components, n, m, n_components);
    result.components.assign(Vt);

    float total_variance = 0.0;
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
            result.scores.set(i, j, U.get(i, j) * S[j]);
        }
    }

    result.reconstruction_error = compute_reconstruction_error(X, U, S, Vt);

    delete[] S;
    return result;
}


int main(int argc, char* argv[]) {
    try {
        std::string input_file = "diabetes_features.csv";
        int n_components = 2;
        bool standardize = false;

        Matrix X = read_csv(input_file);
        if (n_components == -1) {
            n_components = min(X.rows, X.cols);
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

        double error = result.reconstruction_error;
        

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}