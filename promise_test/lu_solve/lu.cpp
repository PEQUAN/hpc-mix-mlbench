#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>

__PROMISE__* create_dense_matrix(int rows, int cols) {
    return new __PROMISE__[rows * cols]();
}

void free_dense_matrix(__PROMISE__* mat) {
    delete[] mat;
}

__PROMISE__* create_vector(int size) {
    return new __PROMISE__[size]();
}

void free_vector(__PROMISE__* vec) {
    delete[] vec;
}

// LU decomposition with partial pivoting
void lu_factorization(const __PROMISE__* A, int n, __PROMISE__* L, __PROMISE__* U, int* P) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U[i * n + j] = A[i * n + j];
        }
        P[i] = i;
        L[i * n + i] = 1.0;
    }

    for (int k = 0; k < n; ++k) {
        __PROMISE__ max_val = abs(U[k * n + k]);
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (abs(U[i * n + k]) > max_val) {
                max_val = abs(U[i * n + k]);
                pivot = i;
            }
        }
        if (abs(max_val) < 1e-15) {
            throw std::runtime_error("Matrix singular or nearly singular");
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
}

// Forward substitution for LU solver
__PROMISE__* lu_forward_substitution(const __PROMISE__* L, int n, const __PROMISE__* b, const int* P) {
    __PROMISE__* y = create_vector(n);
    for (int i = 0; i < n; ++i) {
        __PROMISE__ sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = b[P[i]] - sum;
    }
    return y;
}

// Backward substitution for LU solver
__PROMISE__* lu_backward_substitution(const __PROMISE__* U, int n, const __PROMISE__* y) {
    __PROMISE__* x = create_vector(n);
    for (int i = n - 1; i >= 0; --i) {
        __PROMISE__ sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += U[i * n + j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i * n + i];
    }
    return x;
}

// Matrix-vector multiplication
void matvec(const __PROMISE__* A, int n, const __PROMISE__* x, __PROMISE__* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

// Compute errors
void compute_errors(const __PROMISE__* A, int n, const __PROMISE__* b, const __PROMISE__* x, const __PROMISE__* x_true, __PROMISE__& ferr, __PROMISE__& nbe, __PROMISE__& cbe) {
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
    matvec(A, n, x, Ax);
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
        for (int j = 0; j < n; ++j) {
            row_sum += abs(A[i * n + j]);
        }
        if (row_sum > A_norm) A_norm = row_sum;
    }
    __PROMISE__ b_norm = 0.0;
    for (int i = 0; i < n; ++i) {
        if (abs(b[i]) > b_norm) b_norm = abs(b[i]);
    }
    nbe = norm_r / (A_norm * x_norm + b_norm);

    // Componentwise backward error: max |r_i| / (|A| * |x| + |b|)_i
    cbe = 0.0;
    double zero = 0.0;
    for (int i = 0; i < n; ++i) {
        __PROMISE__ axb = 0.0;
        for (int j = 0; j < n; ++j) {
            axb += abs(A[i * n + j]) * abs(x[j]);
        }
        axb += abs(b[i]);
        __PROMISE__ temp = axb > zero ? abs(r[i]) / axb : zero;
        if (temp > cbe) cbe = temp;
    }

    free_vector(Ax);
    free_vector(r);
}

// Write solution and errors
void write_solution(const __PROMISE__* x, int size, const std::string& filename, __PROMISE__ ferr, __PROMISE__ nbe, __PROMISE__ cbe) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }
    file << "LU Solution\n";
    for (int i = 0; i < size; ++i) {
        file << x[i] << "\n";
    }
    file << "\nErrors\n";
    file << "Forward Error: " << ferr << "\n";
    file << "Normwise Backward Error: " << nbe << "\n";
    file << "Componentwise Backward Error: " << cbe << "\n";
    file.close();
}

// Read Matrix Market file
__PROMISE__* read_matrix_market(const std::string& filename, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening matrix file: " << filename << std::endl;
        return nullptr;
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
        return nullptr;
    }
    n = rows;

    __PROMISE__* A = create_dense_matrix(n, n);
    for (int i = 0; i < nnz; ++i) {
        if (!std::getline(file, line)) {
            std::cerr << "Error reading matrix entries\n";
            free_dense_matrix(A);
            file.close();
            return nullptr;
        }
        int row, col;
        __PROMISE__ val;
        std::istringstream entry(line);
        entry >> row >> col >> val;
        A[(row - 1) * n + (col - 1)] = val;
    }

    file.close();
    return A;
}

int main() {
    int n;
    std::string matrix_file = "1138_bus.mtx";

    __PROMISE__* A = read_matrix_market(matrix_file, n);
    if (!A) {
        std::cerr << "Failed to read matrix A\n";
        return 1;
    }

    __PROMISE__* x_true = create_vector(n);
    for (int i = 0; i < n; ++i) {
        x_true[i] = 1.0;
    }
    __PROMISE__* b = create_vector(n);
    matvec(A, n, x_true, b);

    __PROMISE__* L = create_dense_matrix(n, n);
    __PROMISE__* U = create_dense_matrix(n, n);
    int* P = new int[n];
    try {
        lu_factorization(A, n, L, U, P);
    } catch (const std::exception& e) {
        std::cerr << "LU factorization failed: " << e.what() << "\n";
        free_dense_matrix(A);
        free_dense_matrix(L);
        free_dense_matrix(U);
        delete[] P;
        free_vector(b);
        free_vector(x_true);
        return 1;
    }

    __PROMISE__* y = lu_forward_substitution(L, n, b, P);
    __PROMISE__* x = lu_backward_substitution(U, n, y);

    __PROMISE__ ferr, nbe, cbe;
    compute_errors(A, n, b, x, x_true, ferr, nbe, cbe);

    PROMISE_CHECK_ARRAY(x_true, n);
    std::cout << "LU Solver Results:\n";
    std::cout << "Forward Error: " << ferr << "\n";
    std::cout << "Normwise Backward Error: " << nbe << "\n";
    std::cout << "Componentwise Backward Error: " << cbe << "\n";

    free_dense_matrix(A);
    free_dense_matrix(L);
    free_dense_matrix(U);
    delete[] P;
    free_vector(b);
    free_vector(y);
    free_vector(x);
    free_vector(x_true);

    return 0;
}