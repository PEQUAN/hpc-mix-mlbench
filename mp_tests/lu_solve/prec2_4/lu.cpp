#include <half.hpp>
#include <floatx.hpp>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>

double* create_vector(int size, bool initialize = true) {
    double* vec = initialize ? new double[size]() : new double[size];
    if (!vec) {
        throw std::runtime_error("Failed to allocate vector");
    }
    return vec;
}

void free_vector(double* vec) {
    delete[] vec;
}

int* create_int_vector(int size, bool initialize = true) {
    int* vec = initialize ? new int[size]() : new int[size];
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
    double* val; 
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

    double* temp_val = create_vector(nnz, false);
    int* temp_row = create_int_vector(nnz, false);
    int* temp_col = create_int_vector(nnz, false);

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
    double* val = create_vector(nnz, false);
    int* col_ind = create_int_vector(nnz, false);
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
void sparse_matvec(const SparseMatrix& A, const double* x, double* y) {
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

// Dense LU decomposition with partial pivoting, storing L and U as dense matrices
void dense_lu_factorization(const SparseMatrix& A, double*& L, double*& U, int* P) {
    if (!A.val || !A.col_ind || !A.row_ptr) {
        throw std::runtime_error("Invalid input matrix");
    }
    int n = A.rows;

    for (int i = 0; i < n; ++i) {
        P[i] = i;
    }

    // Convert sparse A to dense format (row-major)
    double* dense_A = create_vector(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            dense_A[i * n + A.col_ind[j]] = A.val[j];
        }
    }

    // Initialize L and U
    L = create_vector(n * n);
    U = create_vector(n * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U[i * n + j] = dense_A[i * n + j];
            L[i * n + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    // Compute matrix infinity norm for singularity check
    flx::floatx<5, 2> A_norm = 0.0;
    for (int i = 0; i < n; ++i) {
        flx::floatx<5, 2> row_sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            row_sum += abs(A.val[j]);
        }
        if (row_sum > A_norm) A_norm = row_sum;
    }

    // LU factorization with partial pivoting
    for (int k = 0; k < n; ++k) {
        flx::floatx<5, 2> max_val = abs(U[k * n + k]);
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (abs(U[i * n + k]) > max_val) {
                max_val = abs(U[i * n + k]);
                pivot = i;
            }
        }
        if (pivot != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(U[k * n + j], U[pivot * n + j]);
                if (j < k) {
                    std::swap(L[k * n + j], L[pivot * n + j]);
                }
            }
            std::swap(P[k], P[pivot]);
        }
        for (int i = k + 1; i < n; ++i) {
            L[i * n + k] = U[i * n + k] / U[k * n + k];
            for (int j = k; j < n; ++j) {
                U[i * n + j] -= L[i * n + k] * U[k * n + j];
            }
        }
    }

    free_vector(dense_A);
}

// Forward substitution for dense L (unit lower triangular)
double* dense_forward_substitution(const double* L, int n, const double* b, const int* P) {
    if (!L || !P) {
        throw std::runtime_error("Invalid inputs in dense_forward_substitution");
    }
    double* y = create_vector(n);
    for (int i = 0; i < n; ++i) {
        flx::floatx<5, 2> sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = b[P[i]] - sum;
    }
    return y;
}

// Backward substitution for dense U (upper triangular)
double* dense_backward_substitution(const double* U, int n, const double* y) {
    if (!U) {
        throw std::runtime_error("Invalid inputs in dense_backward_substitution");
    }
    double* x = create_vector(n);
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += U[i * n + j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i * n + i];
    }
    return x;
}

int main() {
    int n;
    std::string matrix_file = "bp_0.mtx";

    SparseMatrix A = read_matrix_market(matrix_file, n);
    if (!A.val) {
        std::cerr << "Failed to read matrix A\n";
        return 1;
    }

    double* x_true = nullptr;
    double* b = nullptr;
    try {
        x_true = create_vector(n);
        for (int i = 0; i < n; ++i) {
            x_true[i] = 3.0;
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

    double* L = nullptr;
    double  * U = nullptr;
    int* P = nullptr;
    try {
        P = create_int_vector(n);
        dense_lu_factorization(A, L, U, P);
    } catch (const std::exception& e) {
        std::cerr << "Dense LU factorization failed: " << e.what() << "\n";
        free_sparse_matrix(A);
        free_vector(x_true);
        free_vector(b);
        if (P) free_int_vector(P);
        if (L) free_vector(L);
        if (U) free_vector(U);
        return 1;
    }

    double * y = nullptr;
    double * x = nullptr;
    try {
        y = dense_forward_substitution(L, n, b, P);
        x = dense_backward_substitution(U, n, y);
        
    } catch (const std::exception& e) {
        std::cerr << "Dense LU solve failed: " << e.what() << "\n";
        free_sparse_matrix(A);
        free_vector(x_true);
        free_vector(b);
        free_vector(y);
        free_vector(x);
        free_vector(L);
        free_vector(U);
        free_int_vector(P);
        return 1;
    }

    PROMISE_CHECK_ARRAY(x, n);
    
    free_sparse_matrix(A);
    free_vector(L);
    free_vector(U);
    free_int_vector(P);
    free_vector(b);
    free_vector(y);
    free_vector(x);
    free_vector(x_true);

    return 0;
}