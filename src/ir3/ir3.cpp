
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <fstream>

// Allocate a 1D array for row-major matrix
double* create_dense_matrix(int rows, int cols) {
    return new double[rows * cols]();
}

void free_dense_matrix(double* mat) {
    delete[] mat;
}

double* create_vector(int size) {
    return new double[size]();
}

void free_vector(double* vec) {
    delete[] vec;
}

template<typename T>
T get_element(T* matrix, int rows, int cols, int i, int j) {
    return matrix[i * cols + j];
}

template<typename T>
void set_element(T* matrix, int rows, int cols, int i, int j, T value) {
    matrix[i * cols + j] = value;
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
                sum += get_element(A, rowsA, colsA, i, k) * get_element(B, rowsB, colsB, k, j);
            }
            set_element(C, rowsA, colsB, i, j, sum);
        }
    }
    return C;
}

template<typename T>
T* transpose(T* A, int rows, int cols) {
    T* T_mat = create_dense_matrix(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            set_element(T_mat, cols, rows, j, i, get_element(A, rows, cols, i, j));
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
            temp[i] = get_element(A, n, n, i, j);
        }
        for (int k = 0; k < j; ++k) {
            T dot = T(0);
            for (int i = 0; i < n; ++i) {
                dot += get_element(Q, n, n, i, k) * get_element(A, n, n, i, j);
            }
            for (int i = 0; i < n; ++i) {
                temp[i] -= dot * get_element(Q, n, n, i, k);
            }
        }
        T norm = T(0);
        for (int i = 0; i < n; ++i) {
            norm += temp[i] * temp[i];
        }
        norm = std::sqrt(norm);
        if (norm < T(1e-10)) norm = T(1e-10);
        for (int i = 0; i < n; ++i) {
            set_element(Q, n, n, i, j, temp[i] / norm);
        }
    }
    delete[] temp;
    return Q;
}

double compute_condition_number(double* sigma, int p, int kl, int ku, int m, int n) {
    if (kl < m - 1 || ku < n - 1) {
        std::cout << "Warning: Condition number for banded matrix may not be exact.\n";
    }
    double sigma_max = sigma[0];
    double sigma_min = sigma[p - 1];
    for (int i = 1; i < p; ++i) {
        if (sigma[i] > sigma_max) sigma_max = sigma[i];
        if (sigma[i] < sigma_min && sigma[i] > 0) sigma_min = sigma[i];
    }
    return sigma_max / sigma_min;
}

double* gallery_randsvd(int n, double kappa, int mode = 3, int kl = -1, int ku = -1, int method = 0, int random_state = 42) {
    std::mt19937 rng(random_state);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    if (n < 1) {
        std::cerr << "n must be a positive integer\n";
        return nullptr;
    }
    int m = n;
    
    if (kappa < 0) {
        kappa = std::sqrt(1.0 / std::numeric_limits<double>::epsilon());
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
    double* sigma = new double[p]();
    
    if (kappa <= 1) {
        if (m != n) {
            std::cerr << "For kappa <= 1, matrix must be square (m == n)\n";
            delete[] sigma;
            return nullptr;
        }
        double lambda_min = std::abs(kappa);
        double lambda_max = 1.0;
        
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
            std::uniform_real_distribution<double> unif(0.0, 1.0);
            sigma[0] = lambda_max;
            if (p > 1) sigma[p-1] = lambda_min;
            for (int i = 1; i < p-1; ++i) {
                double r = unif(rng);
                sigma[i] = lambda_max * std::exp(std::log(lambda_min / lambda_max) * r);
            }
        }
        
        if (mode < 0) {
            std::sort(sigma, sigma + p);
        } else {
            std::sort(sigma, sigma + p, std::greater<double>());
        }
        
        for (int i = 0; i < p; ++i) {
            if (sigma[i] <= 0) {
                std::cerr << "Eigenvalues must be positive for symmetric positive definite matrix\n";
                delete[] sigma;
                return nullptr;
            }
        }
        
        double* X = create_dense_matrix(n, n);
        for (int i = 0; i < n * n; ++i) {
            X[i] = dist(rng);
        }
        
        double* Q = gram_schmidt(X, n);
        free_dense_matrix(X);
        
        double* D = create_dense_matrix(n, n);
        for (int i = 0; i < n; ++i) {
            set_element(D, n, n, i, i, sigma[i]);
        }
        
        double* QT = transpose(Q, n, n);
        double* temp = matrix_multiply(D, n, n, QT, n, n);
        double* A = matrix_multiply(Q, n, n, temp, n, n);
        
        std::cout << "Condition number: " << compute_condition_number(sigma, p, kl, ku, n, n) << "\n";
        
        free_dense_matrix(Q);
        free_dense_matrix(D);
        free_dense_matrix(QT);
        free_dense_matrix(temp);
        delete[] sigma;
        return A;
    }
    
    if (std::abs(kappa) < 1) {
        std::cerr << "For non-symmetric case, abs(kappa) must be >= 1\n";
        delete[] sigma;
        return nullptr;
    }
    
    double sigma_max = 1.0;
    double sigma_min = sigma_max / std::abs(kappa);
    
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
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        sigma[0] = sigma_max;
        if (p > 1) sigma[p-1] = sigma_min;
        for (int i = 1; i < p-1; ++i) {
            double r = unif(rng);
            sigma[i] = sigma_max * std::exp(std::log(sigma_min / sigma_max) * r);
        }
    }
    
    if (mode < 0) {
        std::sort(sigma, sigma + p);
    } else {
        std::sort(sigma, sigma + p, std::greater<double>());
    }
    
    std::cout << "Generated sigma for mode=" << mode << "\n";
    // for (int i = 0; i < p; ++i) {
    //    std::cout << sigma[i] << " ";
    // }
    std::cout << "\n";
    
    double* Sigma = create_dense_matrix(m, n);
    for (int i = 0; i < p; ++i) {
        set_element(Sigma, m, n, i, i, sigma[i]);
    }
    
    double* X = create_dense_matrix(m, m);
    double* Y = create_dense_matrix(n, n);
    for (int i = 0; i < m * m; ++i) {
        X[i] = dist(rng);
    }
    for (int i = 0; i < n * n; ++i) {
        Y[i] = dist(rng);
    }
    
    double* U = gram_schmidt(X, m);
    double* V = gram_schmidt(Y, n);
    free_dense_matrix(X);
    free_dense_matrix(Y);
    
    double* VT = transpose(V, n, n);
    double* temp = matrix_multiply(Sigma, m, n, VT, n, n);
    double* A = matrix_multiply(U, m, m, temp, m, n);
    
    if (kl < m - 1 || ku < n - 1) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j < i - kl || j > i + ku) {
                    set_element(A, m, n, i, j, 0.0);
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

void matvec(const double* A, int n, const double* x, double* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            y[i] += A[i * n + j] * x[j];
        }
    }
}

void lu_factorization(const double* A, int n, double* L, double* U, int* P) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U[i * n + j] = A[i * n + j];
        }
        P[i] = i;
        L[i * n + i] = 1.0;
    }

    for (int k = 0; k < n; ++k) {
        double max_val = std::abs(U[k * n + k]);
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(U[i * n + k]) > max_val) {
                max_val = std::abs(U[i * n + k]);
                pivot = i;
            }
        }
        if (std::abs(max_val) < 1e-15) {
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

double* forward_substitution(const double* L, int n, const double* b, const int* P) {
    double* y = create_vector(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = b[P[i]] - sum;
    }
    return y;
}

double* backward_substitution(const double* U, int n, const double* y) {
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

double* vec_sub(const double* a, const double* b, int size) {
    double* result = create_vector(size);
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

double* vec_add(const double* a, const double* b, int size) {
    double* result = create_vector(size);
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

double* initial_solve(const double* L, const double* U, int n, const int* P, const double* b) {
    double* y = forward_substitution(L, n, b, P);
    double* x = backward_substitution(U, n, y);
    free_vector(y);
    return x;
}

double* compute_residual(const double* A, int n, const double* b, const double* x) {
    double* Ax = create_vector(n);
    matvec(A, n, x, Ax);
    double* r = vec_sub(b, Ax, n);
    free_vector(Ax);
    return r;
}

double* solve_correction(const double* L, const double* U, int n, const int* P, const double* r) {
    double* y = forward_substitution(L, n, r, P);
    double* d = backward_substitution(U, n, y);
    free_vector(y);
    return d;
}

double* update_solution(const double* x, const double* d, int n) {
    return vec_add(x, d, n);
}

void write_solution(const double* x, int size, const std::string& filename, const double* residual_history, int history_size) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }
    file << "x\n";
    for (int i = 0; i < size; ++i) {
        file << x[i] << "\n";
    }
    file << "\nResidual History\n";
    for (int i = 0; i < history_size; ++i) {
        file << i << "," << residual_history[i] << "\n";
    }
    file.close();
}

double* iterative_refinement(const double* A, int n, const double* b, const double* x_true, double kappa, int max_iter, double*& residual_history, double*& ferr_history, double*& nbe_history, double*& cbe_history, int& history_size) {
    if (n > 10000) {
        std::cerr << "Error: Matrix too large for dense conversion\n";
        return nullptr;
    }

    history_size = 0;
    residual_history = new double[max_iter];
    ferr_history = new double[max_iter];
    nbe_history = new double[max_iter];
    cbe_history = new double[max_iter];

    double* L = create_dense_matrix(n, n);
    double* U = create_dense_matrix(n, n);
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

    double* x = initial_solve(L, U, n, P, b);

    double u = std::numeric_limits<double>::epsilon(); // Machine epsilon

    std::cout << "u:" << u << std::endl;
    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute residual
        double* r = compute_residual(A, n, b, x);

        // Compute residual norm
        double norm_r = 0.0;
        for (int i = 0; i < n; ++i) {
            norm_r += r[i] * r[i];
        }
        norm_r = std::sqrt(norm_r);
        residual_history[history_size] = norm_r;

        // Compute forward error: max |x - x_true| / max |x_true|
        double ferr = 0.0;
        double x_true_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            double err = std::abs(x[i] - x_true[i]);
            if (err > ferr) ferr = err;
            if (std::abs(x_true[i]) > x_true_norm) x_true_norm = std::abs(x_true[i]);
        }
        ferr_history[history_size] = x_true_norm > 0 ? ferr / x_true_norm : ferr;

        // Compute normwise backward error: ||r|| / (||A|| * ||x|| + ||b||)
        double x_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(x[i]) > x_norm) x_norm = std::abs(x[i]);
        }
        double A_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < n; ++j) {
                row_sum += std::abs(A[i * n + j]);
            }
            if (row_sum > A_norm) A_norm = row_sum;
        }
        double b_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(b[i]) > b_norm) b_norm = std::abs(b[i]);
        }
        nbe_history[history_size] = norm_r / (A_norm * x_norm + b_norm);

        // Compute componentwise backward error: max |r_i| / (|A| * |x| + |b|)_i
        double* temp = new double[n];
        double cbe = 0.0;
        for (int i = 0; i < n; ++i) {
            double axb = 0.0;
            for (int j = 0; j < n; ++j) {
                axb += std::abs(A[i * n + j]) * std::abs(x[j]);
            }
            axb += std::abs(b[i]);
            temp[i] = axb > 0 ? std::abs(r[i]) / axb : 0.0;
            if (temp[i] > cbe) cbe = temp[i];
        }
        cbe_history[history_size] = cbe;
        delete[] temp;

        history_size++;

        std::cout << "u * std::sqrt(kappa):" << std::sqrt(u * kappa) << "\n";
        // New stopping criterion: max(max(ferr, nbe), cbe) <= u * kappa
        if (std::max(std::max(ferr_history[iter], nbe_history[iter]), cbe_history[iter]) <= u * kappa) {
            std::cout << "Converged after " << iter + 1 << " iterations\n";
            free_vector(r);
            break;
        }

        std::cout << "Iteration " << iter << ": residual=" << residual_history[iter]
                    << ", ferr=" << ferr_history[iter]
                    << ", nbe=" << nbe_history[iter]
                    << ", cbe=" << cbe_history[iter] << "\n";
                    

        // Solve for correction
        double* d = solve_correction(L, U, n, P, r);
        free_vector(r);

        // Update solution
        double* x_new = update_solution(x, d, n);
        free_vector(d);
        free_vector(x);
        x = x_new;
    }

    free_dense_matrix(L);
    free_dense_matrix(U);
    delete[] P;
    return x;
}

int main() {
    int n = 100; // Matrix size
    double kappa = 1e12; // Condition number
    int max_iter = n;

    // Generate matrix A using gallery_randsvd
    double* A = gallery_randsvd(n, kappa);
    if (!A) {
        std::cerr << "Failed to generate matrix A\n";
        return 1;
    }
    //std::cout << "\nMatrix A (kappa=" << kappa << "):\n";
    //for (int i = 0; i < n; ++i) {
    //    for (int j = 0; j < n; ++j) {
    //        std::cout << std::fixed << std::setprecision(6) << get_element(A, n, n, i, j) << " ";
    //    }
    //    std::cout << "\n";
    //}

    // Generate true solution and right-hand side
    double* x_true = create_vector(n);
    for (int i = 0; i < n; ++i) {
        x_true[i] = 1.0;
    }
    double* b = create_vector(n);
    matvec(A, n, x_true, b);

    // Run iterative refinement
    double* residual_history = nullptr;
    double* ferr_history = nullptr;
    double* nbe_history = nullptr;
    double* cbe_history = nullptr;
    int history_size = 0;
    double* x = iterative_refinement(A, n, b, x_true, kappa, max_iter, residual_history, ferr_history, nbe_history, cbe_history, history_size);

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

    // Write solution to file
    std::string output_file = "solution.txt";
    write_solution(x, n, output_file, residual_history, history_size);
    // Clean up
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