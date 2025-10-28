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
    __PROMISE__* U;    // Left singular vectors (n x k, column-major)
    __PROMISE__* S;    // Singular values (k)
    __PROMISE__* V;    // Right singular vectors (n x k, column-major)
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

    struct Entry { int row, col; __PROMISE__ val; };
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
        __PROMISE__ val;
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
    A.values = new __PROMISE__[entry_count];
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

__PROMISE__* matvec(const CSRMatrix& A, const __PROMISE__* x) {
    __PROMISE__* y = new __PROMISE__[A.n]();
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return y;
}

DenseMatrix matmat(const CSRMatrix& A, const DenseMatrix& X) {
    DenseMatrix Y = {A.n, X.cols, new __PROMISE__[A.n * X.cols]()};
    for (int j = 0; j < X.cols; ++j) {
        __PROMISE__* y = matvec(A, X.data + j * X.rows);
        std::memcpy(Y.data + j * Y.rows, y, A.n * sizeof(__PROMISE__));
        delete[] y;
    }
    return Y;
}

__PROMISE__ dot(const __PROMISE__* a, const __PROMISE__* b, int n) {
    __PROMISE__ sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

__PROMISE__ norm(const __PROMISE__* v, int n) {
    __PROMISE__ d = dot(v, v, n);
    return sqrt(d);
}

DenseMatrix random_matrix(int rows, int cols) {
    DenseMatrix M = {rows, cols, new __PROMISE__[rows * cols]};
    std::mt19937 gen(2025);
    std::normal_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < rows * cols; ++i) {
        M.data[i] = dis(gen);
    }
    return M;
}

DenseMatrix modified_gram_schmidt(const DenseMatrix& A) {
    DenseMatrix Q = {A.rows, A.cols, new __PROMISE__[A.rows * A.cols]};
    std::memcpy(Q.data, A.data, A.rows * A.cols * sizeof(__PROMISE__));

    for (int j = 0; j < A.cols; ++j) {
        __PROMISE__* qj = Q.data + j * A.rows;
        // Normalize first
        __PROMISE__ nrm = norm(qj, A.rows);
        if (nrm < 1e-10) {
            std::cerr << "MGS: Near-zero norm at column " << j << std::endl;
            continue;
        }
        for (int k = 0; k < A.rows; ++k) {
            qj[k] /= nrm;
        }
        // Orthogonalize subsequent vectors
        for (int i = j + 1; i < A.cols; ++i) {
            __PROMISE__* qi = Q.data + i * A.rows;
            __PROMISE__ proj = dot(qi, qj, A.rows);
            for (int k = 0; k < A.rows; ++k) {
                qi[k] -= proj * qj[k];
            }
        }
    }
    return Q;
}

void dense_svd(const DenseMatrix& A, __PROMISE__* U, __PROMISE__* S, __PROMISE__* V) {
    int m = A.rows, n = A.cols;
    DenseMatrix AA = {m, n, new __PROMISE__[m * n]};
    std::memcpy(AA.data, A.data, m * n * sizeof(__PROMISE__));

    DenseMatrix ATA = {n, n, new __PROMISE__[n * n]()};
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < m; ++k) {
                ATA.data[i + j * n] += AA.data[k + i * m] * AA.data[k + j * m];
            }
        }
    }

    DenseMatrix Vmat = {n, n, new __PROMISE__[n * n]()};
    for (int i = 0; i < n; ++i) Vmat.data[i + i * n] = 1.0;

    for (int k = 0; k < n; ++k) {
        __PROMISE__* vk = Vmat.data + k * n;
        for (int iter = 0; iter < 20; ++iter) {
            __PROMISE__* temp = new __PROMISE__[n]();
            // temp = ATA * vk
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    temp[i] += ATA.data[i + j * n] * vk[j];
                }
            }
            __PROMISE__ nrm = norm(temp, n);
            if (nrm < 1e-10) break;
            for (int i = 0; i < n; ++i) vk[i] = temp[i] / nrm;
            delete[] temp;
            for (int j = 0; j < k; ++j) {
                __PROMISE__* vj = Vmat.data + j * n;
                __PROMISE__ proj = dot(vk, vj, n);
                for (int i = 0; i < n; ++i) {
                    vk[i] -= proj * vj[i];
                }
            }
            nrm = norm(vk, n);
            if (nrm < 1e-10) break;
            for (int i = 0; i < n; ++i) vk[i] /= nrm;
        }
        // Compute singular value
        __PROMISE__* Avk = new __PROMISE__[m]();
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

    std::memcpy(V, Vmat.data, n * n * sizeof(__PROMISE__));
    free_dense_matrix(ATA);
    free_dense_matrix(AA);
    free_dense_matrix(Vmat);
}

SVDResult randomized_svd(const CSRMatrix& A, int k, int oversample = 10, int power_iter = 5) {
    int n = A.n;
    int l = k + oversample;

    DenseMatrix Omega = random_matrix(n, l);
    DenseMatrix Y = matmat(A, Omega);

    for (int i = 0; i < power_iter; ++i) {
        DenseMatrix ATY = {n, l, new __PROMISE__[n * l]()};
        for (int j = 0; j < l; ++j) {
            __PROMISE__* yj = Y.data + j * n;
            __PROMISE__* atyj = matvec(A, yj); // Symmetric matrix
            std::memcpy(ATY.data + j * n, atyj, n * sizeof(__PROMISE__));
            delete[] atyj;
        }
        free_dense_matrix(Y);
        Y = matmat(A, ATY);
        free_dense_matrix(ATY);
    }

    DenseMatrix Q = modified_gram_schmidt(Y);
    free_dense_matrix(Y);

    DenseMatrix B = {l, n, new __PROMISE__[l * n]()};
    for (int j = 0; j < n; ++j) {
        __PROMISE__* ej = new __PROMISE__[n]();
        ej[j] = 1.0;
        __PROMISE__* Aej = matvec(A, ej);
        for (int i = 0; i < l; ++i) {
            B.data[i + j * l] = dot(Q.data + i * n, Aej, n);
        }
        delete[] Aej;
        delete[] ej;
    }

    __PROMISE__* Ub = new __PROMISE__[l * k];
    __PROMISE__* Sb = new __PROMISE__[k];
    __PROMISE__* Vb = new __PROMISE__[n * k];
    DenseMatrix Bk = {l, k, new __PROMISE__[l * k]};
    for (int j = 0; j < k; ++j) {
        std::memcpy(Bk.data + j * l, B.data + j * l, l * sizeof(__PROMISE__));
    }
    dense_svd(Bk, Ub, Sb, Vb);
    free_dense_matrix(Bk);
    free_dense_matrix(B);

    __PROMISE__* U = new __PROMISE__[n * k]();
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

int main() {
    std::string filename = "1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) {
        free_csr_matrix(A);
        return 1;
    }

    int k = 100;
    SVDResult result = randomized_svd(A, k, 10, 3);


    __PR_1__* solution = new __PR_1__[k];

    for (int i = 0; i < k; ++i) {
        solution[i] = result.U[i];
    }

    PROMISE_CHECK_ARRAY(solution, k);

    free_csr_matrix(A);
    return 0;
}