#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

struct CSRMatrix {
    int n;           // Number of rows/columns (square matrix)
    double* values;  // Non-zero values
    int* col_indices;// Column indices of non-zeros
    int* row_ptr;    // Row pointers
    int nnz;         // Number of non-zeros
};

struct DenseMatrix {
    int rows;
    int cols;
    double* data; // Column-major order
};

struct SVDResult {
    double* U;    // Left singular vectors (n x k, column-major)
    double* S;    // Singular values (k)
    double* V;    // Right singular vectors (n x k, column-major)
    int k;        // Rank of approximation
};

void free_csr_matrix(CSRMatrix& A) {
    delete[] A.values;
    delete[] A.col_indices;
    delete[] A.row_ptr;
    A.values = nullptr;
    A.col_indices = nullptr;
    A.row_ptr = nullptr;
}

void free_dense_matrix(DenseMatrix& M) {
    delete[] M.data;
    M.data = nullptr;
}

void free_svd_result(SVDResult& result) {
    delete[] result.U;
    delete[] result.S;
    delete[] result.V;
    result.U = nullptr;
    result.S = nullptr;
    result.V = nullptr;
}

CSRMatrix read_mtx_file(const std::string& filename) {
    CSRMatrix A = {0, nullptr, nullptr, nullptr, 0};
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return A;
    }

    std::string line;
    while (getline(file, line) && line[0] == '%') {}

    std::stringstream ss(line);
    int n, m, nz;
    ss >> n >> m >> nz;
    if (n != m) {
        std::cerr << "Error: Matrix must be square" << std::endl;
        return A;
    }
    A.n = n;

    struct Entry { int row, col; double val; };
    Entry* entries = new Entry[2 * nz];
    int entry_count = 0;

    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            delete[] entries;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        double val;
        ss >> i >> j >> val;
        i--; j--;
        entries[entry_count++] = {i, j, val};
        if (i != j) entries[entry_count++] = {j, i, val};
    }

    int* nnz_per_row = new int[n]();
    for (int k = 0; k < entry_count; ++k) {
        nnz_per_row[entries[k].row]++;
    }

    A.nnz = entry_count;
    A.values = new double[entry_count];
    A.col_indices = new int[entry_count];
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row[i];
    }

    std::sort(entries, entries + entry_count, 
        [](const Entry& a, const Entry& b) { 
            return a.row == b.row ? a.col < b.col : a.row < b.row; 
        });

    for (int k = 0; k < entry_count; ++k) {
        A.col_indices[k] = entries[k].col;
        A.values[k] = entries[k].val;
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << entry_count << " non-zeros" << std::endl;

    delete[] nnz_per_row;
    delete[] entries;
    return A;
}

double* matvec(const CSRMatrix& A, const double* x) {
    double* y = new double[A.n]();
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return y;
}

DenseMatrix matmat(const CSRMatrix& A, const DenseMatrix& X) {
    DenseMatrix Y = {A.n, X.cols, new double[A.n * X.cols]()};
    for (int j = 0; j < X.cols; ++j) {
        double* y = matvec(A, X.data + j * X.rows);
        std::memcpy(Y.data + j * Y.rows, y, A.n * sizeof(double));
        delete[] y;
    }
    return Y;
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
    if (std::isnan(d) || std::isinf(d)) return -1.0;
    return std::sqrt(d);
}

DenseMatrix random_matrix(int rows, int cols) {
    DenseMatrix M = {rows, cols, new double[rows * cols]};
    std::mt19937 gen(2025);
    std::normal_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < rows * cols; ++i) {
        M.data[i] = dis(gen);
    }
    return M;
}

DenseMatrix modified_gram_schmidt(const DenseMatrix& A) {
    DenseMatrix Q = {A.rows, A.cols, new double[A.rows * A.cols]};
    std::memcpy(Q.data, A.data, A.rows * A.cols * sizeof(double));

    for (int j = 0; j < A.cols; ++j) {
        double* qj = Q.data + j * A.rows;
        // Normalize first
        double nrm = norm(qj, A.rows);
        if (nrm < 1e-10) {
            std::cerr << "MGS: Near-zero norm at column " << j << std::endl;
            continue;
        }
        for (int k = 0; k < A.rows; ++k) {
            qj[k] /= nrm;
        }
        // Orthogonalize subsequent vectors
        for (int i = j + 1; i < A.cols; ++i) {
            double* qi = Q.data + i * A.rows;
            double proj = dot(qi, qj, A.rows);
            for (int k = 0; k < A.rows; ++k) {
                qi[k] -= proj * qj[k];
            }
        }
    }
    return Q;
}

void dense_svd(const DenseMatrix& A, double* U, double* S, double* V) {
    int m = A.rows, n = A.cols;
    DenseMatrix AA = {m, n, new double[m * n]};
    std::memcpy(AA.data, A.data, m * n * sizeof(double));

    DenseMatrix ATA = {n, n, new double[n * n]()};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < m; ++k) {
                ATA.data[i + j * n] += AA.data[k + i * m] * AA.data[k + j * m];
            }
        }
    }

    DenseMatrix Vmat = {n, n, new double[n * n]()};
    for (int i = 0; i < n; ++i) Vmat.data[i + i * n] = 1.0;

    for (int k = 0; k < n; ++k) {
        double* vk = Vmat.data + k * n;
        for (int iter = 0; iter < 20; ++iter) {
            double* temp = new double[n]();
            // temp = ATA * vk
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    temp[i] += ATA.data[i + j * n] * vk[j];
                }
            }
            double nrm = norm(temp, n);
            if (nrm < 1e-10) break;
            for (int i = 0; i < n; ++i) vk[i] = temp[i] / nrm;
            delete[] temp;
            for (int j = 0; j < k; ++j) {
                double* vj = Vmat.data + j * n;
                double proj = dot(vk, vj, n);
                for (int i = 0; i < n; ++i) {
                    vk[i] -= proj * vj[i];
                }
            }
            nrm = norm(vk, n);
            if (nrm < 1e-10) break;
            for (int i = 0; i < n; ++i) vk[i] /= nrm;
        }
        // Compute singular value
        double* Avk = new double[m]();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                Avk[i] += AA.data[i + j * m] * vk[j];
            }
        }
        S[k] = norm(Avk, m);
        if (S[k] > 1e-10) {
            for (int i = 0; i < m; ++i) {
                U[i + k * m] = Avk[i] / S[k];
            }
        }
        delete[] Avk;
    }

    std::memcpy(V, Vmat.data, n * n * sizeof(double));
    free_dense_matrix(ATA);
    free_dense_matrix(AA);
    free_dense_matrix(Vmat);
}

SVDResult randomized_svd(const CSRMatrix& A, int k, int oversample = 10, int power_iter = 3) {
    int n = A.n;
    int l = k + oversample;

    DenseMatrix Omega = random_matrix(n, l);
    DenseMatrix Y = matmat(A, Omega);

    for (int i = 0; i < power_iter; ++i) {
        DenseMatrix ATY = {n, l, new double[n * l]()};
        for (int j = 0; j < l; ++j) {
            double* yj = Y.data + j * n;
            double* atyj = matvec(A, yj); // Symmetric matrix
            std::memcpy(ATY.data + j * n, atyj, n * sizeof(double));
            delete[] atyj;
        }
        free_dense_matrix(Y);
        Y = matmat(A, ATY);
        free_dense_matrix(ATY);
    }

    DenseMatrix Q = modified_gram_schmidt(Y);
    free_dense_matrix(Y);

    DenseMatrix B = {l, n, new double[l * n]()};
    for (int j = 0; j < n; ++j) {
        double* ej = new double[n]();
        ej[j] = 1.0;
        double* Aej = matvec(A, ej);
        for (int i = 0; i < l; ++i) {
            B.data[i + j * l] = dot(Q.data + i * n, Aej, n);
        }
        delete[] Aej;
        delete[] ej;
    }

    double* Ub = new double[l * k];
    double* Sb = new double[k];
    double* Vb = new double[n * k];
    DenseMatrix Bk = {l, k, new double[l * k]};
    for (int j = 0; j < k; ++j) {
        std::memcpy(Bk.data + j * l, B.data + j * l, l * sizeof(double));
    }
    dense_svd(Bk, Ub, Sb, Vb);
    free_dense_matrix(Bk);
    free_dense_matrix(B);

    double* U = new double[n * k]();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            for (int m = 0; m < l; ++m) {
                U[i + j * n] += Q.data[i + m * n] * Ub[m + j * l];
            }
        }
    }

    SVDResult result = {U, Sb, Vb, k};
    free_dense_matrix(Q);
    free_dense_matrix(Omega);
    delete[] Ub;
    return result;
}

// Accuracy evaluation
double frobenius_norm(const CSRMatrix& A) {
    double sum = 0.0;
    for (int i = 0; i < A.nnz; ++i) {
        sum += A.values[i] * A.values[i];
    }
    return std::sqrt(sum);
}

double frobenius_norm_approx(const SVDResult& result) {
    double sum = 0.0;
    for (int i = 0; i < result.k; ++i) {
        sum += result.S[i] * result.S[i];
    }
    return std::sqrt(sum);
}

double check_orthogonality(const double* X, int n, int k) {
    double max_off_diag = 0.0;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            double dot_prod = 0.0;
            for (int m = 0; m < n; ++m) {
                dot_prod += X[m + i * n] * X[m + j * n];
            }
            if (i == j) {
                max_off_diag = std::max(max_off_diag, std::abs(dot_prod - 1.0));
            } else {
                max_off_diag = std::max(max_off_diag, std::abs(dot_prod));
            }
        }
    }
    return max_off_diag;
}

void evaluate_svd_accuracy(const CSRMatrix& A, const SVDResult& result) {
    double norm_A = frobenius_norm(A);
    std::cout << "Frobenius norm of A: " << norm_A << std::endl;

    double norm_approx = frobenius_norm_approx(result);
    std::cout << "Frobenius norm of U * S * V^T: " << norm_approx << std::endl;

    double error = std::sqrt(std::max(0.0, norm_A * norm_A - norm_approx * norm_approx));
    std::cout << "Approximation error (Frobenius norm): " << error << std::endl;

    double relative_error = error / norm_A;
    std::cout << "Relative error: " << relative_error << std::endl;

    double u_orth_error = check_orthogonality(result.U, A.n, result.k);
    double v_orth_error = check_orthogonality(result.V, A.n, result.k);
    std::cout << "Max orthogonality error in U^T U: " << u_orth_error << std::endl;
    std::cout << "Max orthogonality error in V^T V: " << v_orth_error << std::endl;

    // Direct error check: compute A - U * S * V^T for a few entries
    double sample_error = 0.0;
    int samples = std::min(A.nnz, 100); // Check up to 100 non-zeros
    for (int s = 0; s < samples; ++s) {
        int idx = s * (A.nnz / samples); // Sample evenly
        int i = 0;
        while (A.row_ptr[i + 1] <= idx) i++;
        int j = A.col_indices[idx];
        double a_ij = A.values[idx];
        double approx_ij = 0.0;
        for (int m = 0; m < result.k; ++m) {
            approx_ij += result.U[i + m * A.n] * result.S[m] * result.V[j + m * A.n];
        }
        sample_error += (a_ij - approx_ij) * (a_ij - approx_ij);
    }
    sample_error = std::sqrt(sample_error / samples);
    std::cout << "Sampled approximation error: " << sample_error << std::endl;
}

void write_svd_result(const SVDResult& result, const std::string& prefix) {
    std::ofstream u_file(prefix + "_U.csv");
    u_file << "U\n";
    for (int i = 0; i < result.k; ++i) {
        for (int j = 0; j < result.k; ++j) {
            u_file << result.U[j + i * result.k];
            if (j < result.k - 1) u_file << ",";
        }
        u_file << "\n";
    }
    u_file.close();

    std::ofstream s_file(prefix + "_S.csv");
    s_file << "S\n";
    for (int i = 0; i < result.k; ++i) {
        s_file << result.S[i] << "\n";
    }
    s_file.close();

    std::ofstream v_file(prefix + "_V.csv");
    v_file << "V\n";
    for (int i = 0; i < result.k; ++i) {
        for (int j = 0; j < result.k; ++j) {
            v_file << result.V[j + i * result.k];
            if (j < result.k - 1) v_file << ",";
        }
        v_file << "\n";
    }
    v_file.close();
}

int main() {
    std::string filename = "../data/suitesparse/1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) {
        free_csr_matrix(A);
        return 1;
    }

    int ks[] = {5, 10, 500};
    for (int k : ks) {
        std::cout << "\nRunning Randomized SVD with k = " << k << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        SVDResult result = randomized_svd(A, k, 10, 3);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
        std::cout << "Computed top " << k << " singular values/vectors" << std::endl;
        std::cout << "Computation time: " << duration.count() << " ms" << std::endl;
        std::cout << "Singular values: ";
        for (int i = 0; i < k; ++i) {
            std::cout << result.S[i] << " ";
        }
        std::cout << std::endl;

        evaluate_svd_accuracy(A, result);
        write_svd_result(result, "results/rsvd_k" + std::to_string(k));
        free_svd_result(result);
    }

    free_csr_matrix(A);
    return 0;
}