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

template<typename T>
T* gram_schmidt(T* A, int n) {
    T* Q = create_dense_matrix(n, n);
    T* temp = new T[n];
    
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            temp[i] = A[i * n + j];
        }
        for (int k = 0; k < j; ++k) {
            T dot = T(0);
            for (int i = 0; i < n; ++i) {
                dot += Q[i * n + k] * A[i * n + j];
            }
            for (int i = 0; i < n; ++i) {
                temp[i] -= dot * Q[i * n + k];
            }
        }
        T norm = T(0);
        for (int i = 0; i < n; ++i) {
            norm += temp[i] * temp[i];
        }
        norm = sqrt(norm);
        if (norm < T(1e-10)) norm = T(1e-10);
        for (int i = 0; i < n; ++i) {
            Q[i * n + j] = temp[i] / norm;
        }
    }
    delete[] temp;
    return Q;
}

__PROMISE__ compute_condition_number(__PROMISE__* sigma, int p, int kl, int ku, int m, int n) {
    if (kl < m - 1 || ku < n - 1) {
        std::cout << "Warning: Condition number for banded matrix may not be exact.\n";
    }
    __PROMISE__ sigma_max = sigma[0];
    __PROMISE__ sigma_min = sigma[p - 1];
    for (int i = 1; i < p; ++i) {
        if (sigma[i] > sigma_max) sigma_max = sigma[i];
        if (sigma[i] < sigma_min && sigma[i] > 0) sigma_min = sigma[i];
    }
    return sigma_max / sigma_min;
}

__PROMISE__* gallery_randsvd(int n, __PROMISE__ kappa, int mode = 3, int kl = -1, int ku = -1, int method = 0, int random_state = 42) {
    std::mt19937 rng(random_state);
    std::normal_distribution<__PROMISE__> dist(0.0, 1.0);
    
    if (n < 1) {
        std::cerr << "n must be a positive integer\n";
        return nullptr;
    }
    int m = n;
    
    if (kappa < 0) {
        kappa = sqrt(1.0 / std::numeric_limits<__PROMISE__>::epsilon());
    }
    
    if (kl < 0) kl = m - 1;
    if (ku < 0) ku = kl;
    if (kl >= m || ku >= n || kl < 0 || ku < 0) {
        std::cerr << "kl and ku must be non-negative and less than matrix dimensions\n";
        return nullptr;
    }
    
    if (mode < -5 || mode > 5 || mode == 0) {
        std::cerr << "Mode must be an integer from -5 to -1 or 1 to 5\n";
        return nullptr;
    }
    if (method != 0 && method != 1) {
        std::cerr << "Method must be 0 or 1\n";
        return nullptr;
    }
    if (kl < m - 1 || ku < n - 1) {
        std::cout << "Warning: Banded matrix may not preserve exact singular values.\n";
    }
    
    int p = std::min(m, n);
    __PROMISE__* sigma = new __PROMISE__[p]();
    
    if (kappa <= 1) {
        if (m != n) {
            std::cerr << "For kappa <= 1, matrix must be square (m == n)\n";
            delete[] sigma;
            return nullptr;
        }
        __PROMISE__ lambda_min = abs(kappa);
        __PROMISE__ lambda_max = 1.0;
        
        if (mode == 1 || mode == -1) {
            for (int i = 0; i < p; ++i) sigma[i] = lambda_min;
            sigma[0] = lambda_max;
        } else if (mode == 2 || mode == -2) {
            for (int i = 0; i < p; ++i) sigma[i] = lambda_max;
            sigma[p-1] = lambda_min;
        } else if (mode == 3 || mode == -3) {
            for (int k = 0; k < p; ++k) {
                sigma[k] = lambda_max * std::pow(lambda_min / lambda_max, k / (p > 1 ? p - 1.0 : 1.0));
            }
        } else if (mode == 4 || mode == -4) {
            for (int k = 0; k < p; ++k) {
                sigma[k] = lambda_max - (k / (p > 1 ? p - 1.0 : 1.0)) * (lambda_max - lambda_min);
            }
        } else if (mode == 5 || mode == -5) {
            std::uniform_real_distribution<__PROMISE__> unif(0.0, 1.0);
            sigma[0] = lambda_max;
            if (p > 1) sigma[p-1] = lambda_min;
            for (int i = 1; i < p-1; ++i) {
                __PROMISE__ r = unif(rng);
                sigma[i] = lambda_max * exp(log(lambda_min / lambda_max) * r);
            }
        }
        
        if (mode < 0) {
            std::sort(sigma, sigma + p);
        } else {
            std::sort(sigma, sigma + p, std::greater<__PROMISE__>());
        }
        
        for (int i = 0; i < p; ++i) {
            if (sigma[i] <= 0) {
                std::cerr << "Eigenvalues must be positive for symmetric positive definite matrix\n";
                delete[] sigma;
                return nullptr;
            }
        }
        
        __PROMISE__* X = create_dense_matrix(n, n);
        for (int i = 0; i < n * n; ++i) {
            X[i] = dist(rng);
        }
        
        __PROMISE__* Q = gram_schmidt(X, n);
        free_dense_matrix(X);
        
        __PROMISE__* D = create_dense_matrix(n, n);
        for (int i = 0; i < n; ++i) {
            D[i * n + i] = sigma[i];
        }
        
        __PROMISE__* QT = transpose(Q, n, n);
        __PROMISE__* temp = matrix_multiply(D, n, n, QT, n, n);
        __PROMISE__* A = matrix_multiply(Q, n, n, temp, n, n);
        
        std::cout << "Condition number: " << compute_condition_number(sigma, p, kl, ku, n, n) << "\n";
        
        free_dense_matrix(Q);
        free_dense_matrix(D);
        free_dense_matrix(QT);
        free_dense_matrix(temp);
        delete[] sigma;
        return A;
    }
    
    if (abs(kappa) < 1) {
        std::cerr << "For non-symmetric case, abs(kappa) must be >= 1\n";
        delete[] sigma;
        return nullptr;
    }
    
    __PROMISE__ sigma_max = 1.0;
    __PROMISE__ sigma_min = sigma_max / abs(kappa);
    
    if (mode == 1 || mode == -1) {
        for (int i = 0; i < p; ++i) sigma[i] = sigma_min;
        sigma[0] = sigma_max;

    } else if (mode == 2 || mode == -2) {
        for (int i = 0; i < p; ++i) sigma[i] = sigma_max;
        sigma[p-1] = sigma_min;

    } else if (mode == 3 || mode == -3) {
        for (int k = 0; k < p; ++k) {
            sigma[k] = sigma_max * std::pow(sigma_min / sigma_max, k / (p > 1 ? p - 1.0 : 1.0));
        }

    } else if (mode == 4 || mode == -4) {
        for (int k = 0; k < p; ++k) {
            sigma[k] = sigma_max - (k / (p > 1 ? p - 1.0 : 1.0)) * (sigma_max - sigma_min);
        }

    } else if (mode == 5 || mode == -5) {
        std::uniform_real_distribution<__PROMISE__> unif(0.0, 1.0);
        sigma[0] = sigma_max;
        if (p > 1) sigma[p-1] = sigma_min;
        for (int i = 1; i < p-1; ++i) {
            __PROMISE__ r = unif(rng);
            sigma[i] = sigma_max * exp(log(sigma_min / sigma_max) * r);
        }
    }
    
    if (mode < 0) {
        std::sort(sigma, sigma + p);
    } else {
        std::sort(sigma, sigma + p, std::greater<__PROMISE__>());
    }
    
    std::cout << "Generated sigma for mode=" << mode << "\n";
    std::cout << "\n";
    
    __PROMISE__* Sigma = create_dense_matrix(m, n);
    for (int i = 0; i < p; ++i) {
        Sigma[i * n + i] = sigma[i];
    }
    
    __PROMISE__* X = create_dense_matrix(m, m);
    __PROMISE__* Y = create_dense_matrix(n, n);
    for (int i = 0; i < m * m; ++i) {
        X[i] = dist(rng);
    }
    for (int i = 0; i < n * n; ++i) {
        Y[i] = dist(rng);
    }
    
    __PROMISE__* U = gram_schmidt(X, m);
    __PROMISE__* V = gram_schmidt(Y, n);
    free_dense_matrix(X);
    free_dense_matrix(Y);
    
    __PROMISE__* VT = transpose(V, n, n);
    __PROMISE__* temp = matrix_multiply(Sigma, m, n, VT, n, n);
    __PROMISE__* A = matrix_multiply(U, m, m, temp, m, n);
    
    if (kl < m - 1 || ku < n - 1) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j < i - kl || j > i + ku) {
                    A[i * n + j] = 0.0;
                }
            }
        }
    }
    
    std::cout << "Condition number: " << compute_condition_number(sigma, p, kl, ku, m, n) << "\n";
    
    free_dense_matrix(U);
    free_dense_matrix(V);
    free_dense_matrix(VT);
    free_dense_matrix(Sigma);
    free_dense_matrix(temp);
    delete[] sigma;
    
    return A;
}

void matvec(const __PROMISE__* A, int n, const __PROMISE__* x, __PROMISE__* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

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

__PROMISE__* forward_substitution_init(const __PROMISE__* L, int n, const __PROMISE__* b, const int* P) {
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

__PROMISE__* backward_substitution_init(const __PROMISE__* U, int n, const __PROMISE__* y) {
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

__PROMISE__* forward_substitution(const __PROMISE__* L, int n, const __PROMISE__* b, const int* P) {
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

__PROMISE__* backward_substitution(const __PROMISE__* U, int n, const __PROMISE__* y) {
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

__PROMISE__* vec_sub(const __PROMISE__* a, const __PROMISE__* b, int size) {
    __PROMISE__* result = create_vector(size);
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

__PROMISE__* vec_add(const __PROMISE__* a, const __PROMISE__* b, int size) {
    __PROMISE__* result = create_vector(size);
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

__PROMISE__* initial_solve(const __PROMISE__* L, const __PROMISE__* U, int n, const int* P, const __PROMISE__* b) {
    __PROMISE__* y = forward_substitution_init(L, n, b, P);
    __PROMISE__* x = backward_substitution_init(U, n, y);
    free_vector(y);
    return x;
}

__PROMISE__* compute_residual(const __PROMISE__* A, int n, const __PROMISE__* b, const __PROMISE__* x) {
    __PROMISE__* Ax = create_vector(n);
    matvec(A, n, x, Ax);
    __PROMISE__* r = vec_sub(b, Ax, n);
    free_vector(Ax);
    return r;
}

__PROMISE__* solve_correction(const __PROMISE__* L, const __PROMISE__* U, int n, const int* P, const __PROMISE__* r) {
    __PROMISE__* y = forward_substitution(L, n, r, P);
    __PROMISE__* d = backward_substitution(U, n, y);
    free_vector(y);
    return d;
}

__PROMISE__* update_solution(const __PROMISE__* x, const __PROMISE__* d, int n) {
    return vec_add(x, d, n);
}


__PROMISE__* iterative_refinement(const __PROMISE__* A, int n, const __PROMISE__* b, const __PROMISE__* x_true, __PROMISE__ kappa, __PROMISE__ tol,
        int max_iter, __PROMISE__*& residual_history, __PROMISE__*& ferr_history, __PROMISE__*& nbe_history, __PROMISE__*& cbe_history, int& history_size) {
    if (n > 10000) {
        std::cerr << "Error: Matrix too large for dense conversion\n";
        return nullptr;
    }

    history_size = 0;
    residual_history = new __PROMISE__[max_iter];
    ferr_history = new __PROMISE__[max_iter];
    nbe_history = new __PROMISE__[max_iter];
    cbe_history = new __PROMISE__[max_iter];

    __PROMISE__* L = create_dense_matrix(n, n);
    __PROMISE__* U = create_dense_matrix(n, n);
    int* P = new int[n];
    try {
        lu_factorization(A, n, L, U, P);
    } catch (const std::exception& e) {
        std::cerr << "LU factorization failed: " << e.what() << "\n";
        delete[] P;
        free_dense_matrix(L);
        free_dense_matrix(U);
        return nullptr;
    }

    __PROMISE__* x = initial_solve(L, U, n, P, b);

    __PROMISE__ u = std::numeric_limits<__PROMISE__>::epsilon(); // Machine epsilon

    std::cout << "u: " << u << std::endl;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute residual
        __PROMISE__* r = compute_residual(A, n, b, x);

        // Compute residual norm
        __PROMISE__ norm_r = 0.0;
        for (int i = 0; i < n; ++i) {
            norm_r += r[i] * r[i];
        }
        norm_r = sqrt(norm_r);
        residual_history[history_size] = norm_r;

        // Compute forward error: max |x - x_true| / max |x_true|
        __PROMISE__ ferr = 0.0;
        __PROMISE__ x_true_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            __PROMISE__ err = abs(x[i] - x_true[i]);
            if (err > ferr) ferr = err;
            if (abs(x_true[i]) > x_true_norm) x_true_norm = abs(x_true[i]);
        }
        ferr_history[history_size] = x_true_norm > 0 ? ferr / x_true_norm : ferr;

        // Compute normwise backward error: ||r|| / (||A|| * ||x|| + ||b||)
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
        nbe_history[history_size] = norm_r / (A_norm * x_norm + b_norm);

        // Compute componentwise backward error: max |r_i| / (|A| * |x| + |b|)_i
        __PROMISE__* temp = new __PROMISE__[n];
        __PROMISE__ cbe = 0.0;
        double zero = 0.0;
        for (int i = 0; i < n; ++i) {
            __PROMISE__ axb = 0.0;
            for (int j = 0; j < n; ++j) {
                axb += abs(A[i * n + j]) * abs(x[j]);
            }
            axb += abs(b[i]);
            temp[i] = axb > zero ? abs(r[i]) / axb : zero;
            if (temp[i] > cbe) cbe = temp[i];
        }
        cbe_history[history_size] = cbe;
        delete[] temp;

        history_size++;

        std::cout << "u * sqrt(kappa): " << sqrt(u * kappa) << "\n";
        // Stopping criterion: max(max(ferr, nbe), cbe) <= u * kappa
        if (ferr_history[iter] <= tol) {
            std::cout << "Converged after " << iter + 1 << " iterations\n";
            free_vector(r);
            break;
        }

        std::cout << "Iteration " << iter << ": residual=" << residual_history[iter]
                  << ", ferr=" << ferr_history[iter]
                  << ", nbe=" << nbe_history[iter]
                  << ", cbe=" << cbe_history[iter] << "\n";

        // Solve for correction
        __PROMISE__* d = solve_correction(L, U, n, P, r);
        free_vector(r);

        // Update solution
        __PROMISE__* x_new = update_solution(x, d, n);
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
__PROMISE__* read_matrix_market(const std::string& filename, int& n) {
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
    __PROMISE__* A = create_dense_matrix(n, n);

    // Read non-zero entries
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
        // Matrix Market uses 1-based indexing, convert to 0-based
        A[(row - 1) * n + (col - 1)] = val;
    }

    file.close();
    return A;
}

int main() {
    int n;
    __PROMISE__ tol = 1e-8;
    std::string matrix_file = "1138_bus.mtx";
    __PROMISE__ kappa = 1e6; // Condition number (approximate or estimated)
    
    __PROMISE__* A = read_matrix_market(matrix_file, n);
    if (!A) {
        std::cerr << "Failed to read matrix A\n";
        return 1;
    }

    int max_iter = n; 
    __PROMISE__* x_true = create_vector(n);
    for (int i = 0; i < n; ++i) {
        x_true[i] = 1.0;
    }
    __PROMISE__* b = create_vector(n);
    matvec(A, n, x_true, b);

    double* residual_history = nullptr;
    double* ferr_history = nullptr;
    double* nbe_history = nullptr;
    double* cbe_history = nullptr;
    int history_size = 0;
    __PROMISE__* x = iterative_refinement(A, n, b, x_true, kappa, tol, max_iter, residual_history, ferr_history, nbe_history, cbe_history, history_size);

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

    double zero = 0.0;
    double final_ferr = history_size > zero ? ferr_history[history_size - 1] : zero;
    double final_nbe = history_size > zero ? nbe_history[history_size - 1] : zero;
    double final_cbe = history_size > zero ? cbe_history[history_size - 1] : zero;

    std::cout << "Final Forward Error: " << final_ferr << "\n";
    std::cout << "Final Normwise Backward Error: " << final_nbe << "\n";
    std::cout << "Final Componentwise Backward Error: " << final_cbe << "\n";

    PROMISE_CHECK_ARRAY(x, n);
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