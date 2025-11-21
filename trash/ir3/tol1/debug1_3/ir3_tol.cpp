#include <half.hpp>
#include <floatx.hpp>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>

float* create_dense_matrix(int rows, int cols) {
    return new float[rows * cols]();
}

void free_dense_matrix(float* mat) {
    delete[] mat;
}

float* create_vector(int size) {
    return new float[size]();
}

void free_vector(float* vec) {
    delete[] vec;
}

template<typename T>
T* matrix_multiply(T* A, int rowsA, int colsA, T* B, int rowsB, int colsB) {
    if (colsA != rowsB) {
        std::cerr << "Matrix dimensions incompatible\n";
        return nullptr;
    }
    T* C = create_dense_matrix(rowsA, colsB);
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            T sum = T(0);
            for (int k = 0; k < colsA; ++k) {
                sum += A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = sum;
        }
    }
    return C;
}

template<typename T>
T* transpose(T* A, int rows, int cols) {
    T* T_mat = create_dense_matrix(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            T_mat[j * rows + i] = A[i * cols + j];
        }
    }
    return T_mat;
}

void matvec(float* A, int n, float* x, float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

void lu_factorization(float* A, int n, float* L, float* U, int* P) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U[i * n + j] = A[i * n + j];
        }
        P[i] = i;
        L[i * n + i] = 1.0;
    }

    for (int k = 0; k < n; ++k) {
        float max_val = abs(U[k * n + k]);
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

float* forward_substitution_init(float* L, int n, float* b, int* P) {
    float* y = create_vector(n);
    for (int i = 0; i < n; ++i) {
        float sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = b[P[i]] - sum;
    }
    return y;
}

float* backward_substitution_init(float* U, int n, float* y) {
    float* x = create_vector(n);
    for (int i = n - 1; i >= 0; --i) {
        float sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += U[i * n + j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i * n + i];
    }
    return x;
}

float* forward_substitution(float* L, int n, float* b, int* P) {
    float* y = create_vector(n);
    for (int i = 0; i < n; ++i) {
        float sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = b[P[i]] - sum;
    }
    return y;
}

float* backward_substitution(float* U, int n, float* y) {
    float* x = create_vector(n);
    for (int i = n - 1; i >= 0; --i) {
        float sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += U[i * n + j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i * n + i];
    }
    return x;
}

float* vec_sub(float* a, float* b, int size) {
    float* result = create_vector(size);
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

float* vec_add(float* a, float* b, int size) {
    float* result = create_vector(size);
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

float* initial_solve(float* L, float* U, int n, int* P, float* b) {
    float* y = forward_substitution_init(L, n, b, P);
    float* x = backward_substitution_init(U, n, y);
    free_vector(y);
    return x;
}

float* compute_residual(float* A, int n, float* b, float* x) {
    float* Ax = create_vector(n);
    matvec(A, n, x, Ax);
    float* r = vec_sub(b, Ax, n);
    free_vector(Ax);
    return r;
}

float* solve_correction(float* L, float* U, int n, int* P, float* r) {
    float* y = forward_substitution(L, n, r, P);
    float* d = backward_substitution(U, n, y);
    free_vector(y);
    return d;
}

float* update_solution(float* x, float* d, int n) {
    return vec_add(x, d, n);
}

float* iterative_refinement(float* A, int n, float* b, float* x_true, float kappa, float tol, int max_iter, double*& residual_history, double*& ferr_history, double*& nbe_history, double*& cbe_history, int& history_size) {
    if (n > 10000) {
        std::cerr << "Error: Matrix too large for dense conversion\n";
        return nullptr;
    }

    history_size = 0;
    residual_history = new double[max_iter];
    ferr_history = new double[max_iter];
    nbe_history = new double[max_iter];
    cbe_history = new double[max_iter];

    float* L = create_dense_matrix(n, n);
    float* U = create_dense_matrix(n, n);
    int* P = new int[n];
    try {
        lu_factorization(A, n, L, U, P);
    } catch (std::exception& e) {
        std::cerr << "LU factorization failed: " << e.what() << "\n";
        delete[] P;
        free_dense_matrix(L);
        free_dense_matrix(U);
        return nullptr;
    }

    float* x = initial_solve(L, U, n, P, b);

    double u = 2.22045e-16; // Machine epsilon
    float prev_dx_norm_inf = 9999999.0; // Previous correction norm

    std::cout << "u: " << u << std::endl;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute residual
        float* r = compute_residual(A, n, b, x);

        // Compute residual norm
        float norm_r = 0.0;
        for (int i = 0; i < n; ++i) {
            norm_r += r[i] * r[i];
        }
        norm_r = sqrt(norm_r);
        residual_history[history_size] = norm_r;

        // Compute forward error: max |x - x_true| / max |x_true|
        float ferr = 0.0;
        float x_true_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            float err = abs(x[i] - x_true[i]);
            if (err > ferr) ferr = err;
            if (abs(x_true[i]) > x_true_norm) x_true_norm = abs(x_true[i]);
        }
        ferr_history[history_size] = x_true_norm > 0 ? ferr / x_true_norm : ferr;

        // Compute normwise backward error: ||r|| / (||A|| * ||x|| + ||b||)
        float x_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (abs(x[i]) > x_norm) x_norm = abs(x[i]);
        }
        float A_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            float row_sum = 0.0;
            for (int j = 0; j < n; ++j) {
                row_sum += abs(A[i * n + j]);
            }
            if (row_sum > A_norm) A_norm = row_sum;
        }
        float b_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (abs(b[i]) > b_norm) b_norm = abs(b[i]);
        }
        nbe_history[history_size] = norm_r / (A_norm * x_norm + b_norm);

        // Compute componentwise backward error: max |r_i| / (|A| * |x| + |b|)_i
        float* temp = new float[n];
        double zeros = 0.0;
        float cbe = 0.0;
        for (int i = 0; i < n; ++i) {
            float axb = 0.0;
            for (int j = 0; j < n; ++j) {
                axb += abs(A[i * n + j]) * abs(x[j]);
            }
            axb += abs(b[i]);
            temp[i] = axb > zeros ? abs(r[i]) / axb : zeros;
            if (temp[i] > cbe) cbe = temp[i];
        }
        cbe_history[history_size] = cbe;
        delete[] temp;

        // Solve for correction
        float* d = solve_correction(L, U, n, P, r);
        free_vector(r);

        // Compute infinity norms for stopping criteria
        float dx_norm_inf = 0.0;
        for (int i = 0; i < n; ++i) {
            if (abs(d[i]) > dx_norm_inf) dx_norm_inf = abs(d[i]);
        }
        float x_norm_inf = 0.0;
        for (int i = 0; i < n; ++i) {
            if (abs(x[i]) > x_norm_inf) x_norm_inf = abs(x[i]);
        }

        // Check stopping criteria
        bool stop = false;
        
        if (x_norm_inf > 0 && dx_norm_inf / x_norm_inf <= u) {
            std::cout << "Converged after " << iter + 1 << " iterations: ||dx||_inf / ||x||_inf <= u\n";
            stop = true;
        } else if (prev_dx_norm_inf > zeros && dx_norm_inf / prev_dx_norm_inf >= tol) {
            std::cout << "Converged after " << iter + 1 << " iterations: ||dx^{(i+1)}||_inf / ||dx^{(i)}||_inf >= tol\n";
            stop = true;
        } else if (iter == max_iter - 1) {
            std::cout << "Stopped after " << max_iter << " iterations: maximum iterations reached\n";
            stop = true;
        }
        
        // Log iteration details
        std::cout << "Iteration " << iter << ": residual=" << residual_history[history_size]
                  << ", ferr=" << ferr_history[history_size]
                  << ", nbe=" << nbe_history[history_size]
                  << ", cbe=" << cbe_history[history_size]
                  << ", ||dx||_inf/||x||_inf=" << (x_norm_inf > zeros ? dx_norm_inf / x_norm_inf : zeros)
                  << ", ||dx^{(i+1)}||_inf/||dx^{(i)}||_inf=" << (prev_dx_norm_inf > zeros ? dx_norm_inf / prev_dx_norm_inf : zeros) << "\n";

        history_size++;
        prev_dx_norm_inf = dx_norm_inf;

        if (stop) {
            free_vector(d);
            break;
        }

        // Update solution
        float* x_new = update_solution(x, d, n);
        free_vector(d);
        free_vector(x);
        x = x_new;
    }

    free_dense_matrix(L);
    free_dense_matrix(U);
    delete[] P;
    return x;
}

// Function to read Matrix Market file and convert to dense matrix
float* read_matrix_market(std::string& filename, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening matrix file: " << filename << std::endl;
        return nullptr;
    }

    std::string line;
    // Skip header comments
    while (std::getline(file, line)) {
        if (line[0] != '%') break;
    }

    // Read matrix dimensions and number of non-zeros
    int rows, cols, nnz;
    std::istringstream iss(line);
    iss >> rows >> cols >> nnz;

    if (rows != cols) {
        std::cerr << "Matrix must be square for this implementation\n";
        file.close();
        return nullptr;
    }
    n = rows;

    // Allocate dense matrix
    float* A = create_dense_matrix(n, n);

    // Read non-zero entries
    for (int i = 0; i < nnz; ++i) {
        if (!std::getline(file, line)) {
            std::cerr << "Error reading matrix entries\n";
            free_dense_matrix(A);
            file.close();
            return nullptr;
        }
        int row, col;
        half_float::half val;
        std::istringstream entry(line);
        entry >> row >> col >> val;
        // Matrix Market uses 1-based indexing, convert to 0-based
        A[(row - 1) * n + (col - 1)] = val;
    }

    file.close();
    return A;
}

int main() {
    int n;
    std::string matrix_file = "1138_bus.mtx";
    double kappa = 1e6; // Condition number (approximate or estimated)
    double tol = 1e-6;   // Tolerance for convergence slowing criterion
    
    // Read matrix A from file
    float* A = read_matrix_market(matrix_file, n);
    int max_iter = n; // Adjusted for larger matrix
    
    if (!A) {
        std::cerr << "Failed to read matrix A\n";
        return 1;
    }

    float* x_true = create_vector(n);
    for (int i = 0; i < n; ++i) {
        x_true[i] = 1.0;
    }
    float* b = create_vector(n);
    matvec(A, n, x_true, b);

    double* residual_history = nullptr;
    double* ferr_history = nullptr;
    double* nbe_history = nullptr;
    double* cbe_history = nullptr;
    int history_size = 0;
    float* x = iterative_refinement(A, n, b, x_true, kappa, tol, max_iter, residual_history, ferr_history, nbe_history, cbe_history, history_size);

    PROMISE_CHECK_ARRAY(x, n);

    if (x == nullptr) {
        std::cerr << "Failed to solve system\n";
        free_dense_matrix(A);
        free_vector(b);
        free_vector(x_true);
        delete[] residual_history;
        delete[] ferr_history;
        delete[] nbe_history;
        delete[] cbe_history;
        return 1;
    }

    double zeros = 0.0;
    double final_ferr = history_size > zeros ? ferr_history[history_size - 1] : zeros;
    double final_nbe = history_size > zeros ? nbe_history[history_size - 1] : zeros;
    double final_cbe = history_size > zeros ? cbe_history[history_size - 1] : zeros;


    std::cout << "Final Forward Error: " << final_ferr << "\n";
    std::cout << "Final Normwise Backward Error: " << final_nbe << "\n";
    std::cout << "Final Componentwise Backward Error: " << final_cbe << "\n";

    free_dense_matrix(A);
    free_vector(b);
    free_vector(x_true);
    free_vector(x);
    delete[] residual_history;
    delete[] ferr_history;
    delete[] nbe_history;
    delete[] cbe_history;
    return 0;
}