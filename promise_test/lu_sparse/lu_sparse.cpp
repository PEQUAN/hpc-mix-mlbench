#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>

__PROMISE__* create_vector(int size) {
    __PROMISE__* vec = new __PROMISE__[size]();
    if (!vec) {
        throw std::runtime_error("Failed to allocate vector");
    }
    return vec;
}

void free_vector(__PROMISE__* vec) {
    delete[] vec;
}

int* create_int_vector(int size) {
    int* vec = new int[size]();
    if (!vec) {
        throw std::runtime_error("Failed to allocate int vector");
    }
    return vec;
}

void free_int_vector(int* vec) {
    delete[] vec;
}

struct SparseMatrix {
// Sparse matrix in CSR format
    __PROMISE__* val; 
    int* col_ind; 
    int* row_ptr; 
    int rows; 
    int cols; 
    int nnz; 
};

void free_sparse_matrix(SparseMatrix& mat) {
    if (mat.val) delete[] mat.val;
    if (mat.col_ind) delete[] mat.col_ind;
    if (mat.row_ptr) delete[] mat.row_ptr;
    mat.val = nullptr;
    mat.col_ind = nullptr;
    mat.row_ptr = nullptr;
    mat.nnz = 0;
}

SparseMatrix read_matrix_market(const std::string& filename, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening matrix file: " << filename << std::endl;
        return {nullptr, nullptr, nullptr, 0, 0, 0};
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    int rows, cols, nnz;
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;

    if (rows != cols) {
        std::cerr << "Matrix must be square for this implementation\n";
        file.close();
        return {nullptr, nullptr, nullptr, 0, 0, 0};
    }
    n = rows;

    __PROMISE__* temp_val = create_vector(nnz); // Temporary storage for coordinate format
    int* temp_row = create_int_vector(nnz);
    int* temp_col = create_int_vector(nnz);

    for (int i = 0; i < nnz; ++i) {
        if (!std::getline(file, line)) {
            std::cerr << "Error reading matrix entries\n";
            free_vector(temp_val);
            free_int_vector(temp_row);
            free_int_vector(temp_col);
            file.close();
            return {nullptr, nullptr, nullptr, 0, 0, 0};
        }
        std::istringstream entry(line);
        entry >> temp_row[i] >> temp_col[i] >> temp_val[i];
        temp_row[i]--; // Convert to 0-based indexing
        temp_col[i]--;
    }
    file.close();

    // Convert to CSR format
    __PROMISE__* val = create_vector(nnz);
    int* col_ind = create_int_vector(nnz);
    int* row_ptr = create_int_vector(rows + 1);

    // Count non-zeros per row
    for (int i = 0; i < nnz; ++i) {
        row_ptr[temp_row[i] + 1]++;
    }
    // Cumulative sum for row_ptr
    for (int i = 1; i <= rows; ++i) {
        row_ptr[i] += row_ptr[i - 1];
    }
    // Place entries in CSR format
    int* row_counts = create_int_vector(rows);
    for (int i = 0; i < nnz; ++i) {
        int r = temp_row[i];
        int pos = row_ptr[r] + row_counts[r];
        val[pos] = temp_val[i];
        col_ind[pos] = temp_col[i];
        row_counts[r]++;
    }

    free_vector(temp_val);
    free_int_vector(temp_row);
    free_int_vector(temp_col);
    free_int_vector(row_counts);

    return {val, col_ind, row_ptr, rows, cols, nnz};
}

// Sparse matrix-vector multiplication
void sparse_matvec(const SparseMatrix& A, const __PROMISE__* x, __PROMISE__* y) {
    if (!A.val || !A.col_ind || !A.row_ptr) {
        throw std::runtime_error("Invalid sparse matrix in sparse_matvec");
    }
    for (int i = 0; i < A.rows; ++i) {
        y[i] = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.val[j] * x[A.col_ind[j]];
        }
    }
}

// Sparse LU decomposition with partial pivoting
void sparse_lu_factorization(const SparseMatrix& A, SparseMatrix& L, SparseMatrix& U, int* P) {
    if (!A.val || !A.col_ind || !A.row_ptr) {
        throw std::runtime_error("Invalid input matrix");
    }
    int n = A.rows;
    L.rows = n;
    L.cols = n;
    U.rows = n;
    U.cols = n;

    for (int i = 0; i < n; ++i) {
        P[i] = i;
    }

    __PROMISE__* dense_A = create_vector(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            dense_A[i * n + A.col_ind[j]] = A.val[j];
        }
    }

    __PROMISE__* dense_L = create_vector(n * n);
    __PROMISE__* dense_U = create_vector(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            dense_U[i * n + j] = dense_A[i * n + j];
        }
        dense_L[i * n + i] = 1.0;
    }

    // LU factorization with partial pivoting
    for (int k = 0; k < n; ++k) {
        __PROMISE__ max_val = abs(dense_U[k * n + k]);
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (abs(dense_U[i * n + k]) > max_val) {
                max_val = abs(dense_U[i * n + k]);
                pivot = i;
            }
        }
        if (abs(max_val) < 1e-15) {
            throw std::runtime_error("Matrix singular or nearly singular");
        }
        if (pivot != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(dense_U[k * n + j], dense_U[pivot * n + j]);
                if (j < k) {
                    std::swap(dense_L[k * n + j], dense_L[pivot * n + j]);
                }
            }
            std::swap(P[k], P[pivot]);
        }
        for (int i = k + 1; i < n; ++i) {
            dense_L[i * n + k] = dense_U[i * n + k] / dense_U[k * n + k];
            for (int j = k; j < n; ++j) {
                dense_U[i * n + j] -= dense_L[i * n + k] * dense_U[k * n + j];
            }
        }
    }

    // Convert L and U to sparse format
    int nnz_L = 0, nnz_U = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (abs(dense_L[i * n + j]) > 1e-15) nnz_L++;
        }
        for (int j = i; j < n; ++j) {
            if (abs(dense_U[i * n + j]) > 1e-15) nnz_U++;
        }
    }

    L.val = create_vector(nnz_L);
    L.col_ind = create_int_vector(nnz_L);
    L.row_ptr = create_int_vector(n + 1);
    U.val = create_vector(nnz_U);
    U.col_ind = create_int_vector(nnz_U);
    U.row_ptr = create_int_vector(n + 1);

    int pos_L = 0, pos_U = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (abs(dense_L[i * n + j]) > 1e-15) {
                L.val[pos_L] = dense_L[i * n + j];
                L.col_ind[pos_L] = j;
                pos_L++;
            }
        }
        L.row_ptr[i + 1] = pos_L;
        for (int j = i; j < n; ++j) {
            if (abs(dense_U[i * n + j]) > 1e-15) {
                U.val[pos_U] = dense_U[i * n + j];
                U.col_ind[pos_U] = j;
                pos_U++;
            }
        }
        U.row_ptr[i + 1] = pos_U;
    }
    L.nnz = nnz_L;
    U.nnz = nnz_U;

    free_vector(dense_A);
    free_vector(dense_L);
    free_vector(dense_U);
}

// Forward substitution for sparse L
__PROMISE__* sparse_forward_substitution(const SparseMatrix& L, int n, const __PROMISE__* b, const int* P) {
    if (!L.val || !L.col_ind || !L.row_ptr || !P) {
        throw std::runtime_error("Invalid inputs in sparse_forward_substitution");
    }
    __PROMISE__* y = create_vector(n);
    for (int i = 0; i < n; ++i) {
        __PROMISE__ sum = 0.0;
        for (int j = L.row_ptr[i]; j < L.row_ptr[i + 1]; ++j) {
            if (L.col_ind[j] < i) {
                sum += L.val[j] * y[L.col_ind[j]];
            }
        }
        y[i] = b[P[i]] - sum;
    }
    return y;
}

// Backward substitution for sparse U
__PROMISE__* sparse_backward_substitution(const SparseMatrix& U, int n, const __PROMISE__* y) {
    if (!U.val || !U.col_ind || !U.row_ptr) {
        throw std::runtime_error("Invalid inputs in sparse_backward_substitution");
    }
    __PROMISE__* x = create_vector(n);
    for (int i = n - 1; i >= 0; --i) {
        __PROMISE__ sum = 0.0;
        __PROMISE__ diag = 0.0;
        for (int j = U.row_ptr[i]; j < U.row_ptr[i + 1]; ++j) {
            if (U.col_ind[j] == i) {
                diag = U.val[j];
            } else if (U.col_ind[j] > i) {
                sum += U.val[j] * x[U.col_ind[j]];
            }
        }
        if (abs(diag) < 1e-15) {
            throw std::runtime_error("U is singular or nearly singular");
        }
        x[i] = (y[i] - sum) / diag;
    }
    return x;
}

// Compute errors
void compute_errors(const SparseMatrix& A, int n, const __PROMISE__* b, const __PROMISE__* x, const __PROMISE__* x_true, __PROMISE__& ferr, __PROMISE__& nbe, __PROMISE__& cbe) {
    if (!A.val || !A.col_ind || !A.row_ptr) {
        throw std::runtime_error("Invalid sparse matrix in compute_errors");
    }
    // Forward error: max |x - x_true| / max |x_true|
    __PROMISE__ x_true_norm = 0.0;
    ferr = 0.0;
    for (int i = 0; i < n; ++i) {
        __PROMISE__ err = abs(x[i] - x_true[i]);
        if (err > ferr) ferr = err;
        if (abs(x_true[i]) > x_true_norm) x_true_norm = abs(x_true[i]);
    }
    ferr = x_true_norm > 0 ? ferr / x_true_norm : ferr;

    // Compute residual: r = b - Ax
    __PROMISE__* Ax = create_vector(n);
    sparse_matvec(A, x, Ax);
    __PROMISE__* r = create_vector(n);
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - Ax[i];
    }

    // Normwise backward error: ||r|| / (||A|| * ||x|| + ||b||)
    __PROMISE__ norm_r = 0.0;
    for (int i = 0; i < n; ++i) {
        norm_r += r[i] * r[i];
    }
    norm_r = sqrt(norm_r);
    __PROMISE__ x_norm = 0.0;
    for (int i = 0; i < n; ++i) {
        if (abs(x[i]) > x_norm) x_norm = abs(x[i]);
    }
    __PROMISE__ A_norm = 0.0;
    for (int i = 0; i < n; ++i) {
        __PROMISE__ row_sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            row_sum += abs(A.val[j]);
        }
        if (row_sum > A_norm) A_norm = row_sum;
    }
    __PROMISE__ b_norm = 0.0;
    for (int i = 0; i < n; ++i) {
        if (abs(b[i]) > b_norm) b_norm = abs(b[i]);
    }
    nbe = (A_norm * x_norm + b_norm) > 0 ? norm_r / (A_norm * x_norm + b_norm) : norm_r;

    // Componentwise backward error: max |r_i| / (|A| * |x| + |b|)_i
    cbe = 0.0;
    double zero = 0.0;
    for (int i = 0; i < n; ++i) {
        __PROMISE__ axb = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            axb += abs(A.val[j]) * abs(x[A.col_ind[j]]);
        }
        axb += abs(b[i]);
        __PROMISE__ temp = axb > zero ? abs(r[i]) / axb : zero;
        if (temp > cbe) cbe = temp;
    }

    free_vector(Ax);
    free_vector(r);
}


int main() {
    int n;
    std::string matrix_file = "1138_bus.mtx";

    SparseMatrix A = read_matrix_market(matrix_file, n);
    if (!A.val) {
        std::cerr << "Failed to read matrix A\n";
        return 1;
    }

    __PROMISE__* x_true = nullptr;
    __PROMISE__* b = nullptr;
    try {
        x_true = create_vector(n);
        for (int i = 0; i < n; ++i) {
            x_true[i] = 1.0;
        }
        b = create_vector(n);
        sparse_matvec(A, x_true, b);
    } catch (const std::exception& e) {
        std::cerr << "Error setting up true solution: " << e.what() << "\n";
        free_sparse_matrix(A);
        if (x_true) free_vector(x_true);
        if (b) free_vector(b);
        return 1;
    }

    SparseMatrix L = {nullptr, nullptr, nullptr, 0, 0, 0};
    SparseMatrix U = {nullptr, nullptr, nullptr, 0, 0, 0};
    int* P = nullptr;
    try {
        P = create_int_vector(n);
        sparse_lu_factorization(A, L, U, P);
    } catch (const std::exception& e) {
        std::cerr << "Sparse LU factorization failed: " << e.what() << "\n";
        free_sparse_matrix(A);
        free_sparse_matrix(L);
        free_sparse_matrix(U);
        free_vector(x_true);
        free_vector(b);
        if (P) free_int_vector(P);
        return 1;
    }

    __PROMISE__* y = nullptr;
    __PROMISE__* x = nullptr;
    try {
        y = sparse_forward_substitution(L, n, b, P);
        x = sparse_backward_substitution(U, n, y);
    } catch (const std::exception& e) {
        std::cerr << "Sparse LU solve failed: " << e.what() << "\n";
        free_sparse_matrix(A);
        free_sparse_matrix(L);
        free_sparse_matrix(U);
        free_vector(x_true);
        free_vector(b);
        free_vector(y);
        if (P) free_int_vector(P);
        return 1;
    }

    PROMISE_CHECK_ARRAY(x, n);

    __PROMISE__ ferr, nbe, cbe;
    try {
        compute_errors(A, n, b, x, x_true, ferr, nbe, cbe);
    } catch (const std::exception& e) {
        std::cerr << "Error computing errors: " << e.what() << "\n";
        free_sparse_matrix(A);
        free_sparse_matrix(L);
        free_sparse_matrix(U);
        free_vector(x_true);
        free_vector(b);
        free_vector(y);
        free_vector(x);
        free_int_vector(P);
        return 1;
    }


    std::cout << "Sparse LU Solver Results:\n";
    std::cout << "Forward Error: " << ferr << "\n";
    std::cout << "Normwise Backward Error: " << nbe << "\n";
    std::cout << "Componentwise Backward Error: " << cbe << "\n";

    free_sparse_matrix(A);
    free_sparse_matrix(L);
    free_sparse_matrix(U);
    free_int_vector(P);
    free_vector(b);
    free_vector(y);
    free_vector(x);
    free_vector(x_true);

    return 0;
}