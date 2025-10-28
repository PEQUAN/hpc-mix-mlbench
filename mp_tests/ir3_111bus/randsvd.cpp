#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>

// Allocate a 1D array for row-major matrix
double* allocate_matrix(int rows, int cols) {
    return new double[rows * cols]();
}

// Deallocate a 1D array
void deallocate_matrix(double* matrix) {
    delete[] matrix;
}

// Get element in row-major order
double get_element(double* matrix, int rows, int cols, int i, int j) {
    return matrix[i * cols + j];
}

// Set element in row-major order
void set_element(double* matrix, int rows, int cols, int i, int j, double value) {
    matrix[i * cols + j] = value;
}

// Matrix multiplication: C = A * B
double* matrix_multiply(double* A, int rowsA, int colsA, double* B, int rowsB, int colsB) {
    if (colsA != rowsB) {
        std::cerr << "Matrix dimensions incompatible for multiplication\n";
        return nullptr;
    }
    double* C = allocate_matrix(rowsA, colsB);
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            double sum = 0.0;
            for (int k = 0; k < colsA; ++k) {
                sum += get_element(A, rowsA, colsA, i, k) * get_element(B, rowsB, colsB, k, j);
            }
            set_element(C, rowsA, colsB, i, j, sum);
        }
    }
    return C;
}

// Transpose a matrix
double* transpose(double* A, int rows, int cols) {
    double* T = allocate_matrix(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            set_element(T, cols, rows, j, i, get_element(A, rows, cols, i, j));
        }
    }
    return T;
}

// Basic Gram-Schmidt QR decomposition (returns Q)
double* gram_schmidt(double* A, int n) {
    double* Q = allocate_matrix(n, n);
    double* temp = new double[n];
    
    for (int j = 0; j < n; ++j) {
        // Copy column j to temp
        for (int i = 0; i < n; ++i) {
            temp[i] = get_element(A, n, n, i, j);
        }
        // Orthogonalize against previous columns
        for (int k = 0; k < j; ++k) {
            double dot = 0.0;
            for (int i = 0; i < n; ++i) {
                dot += get_element(Q, n, n, i, k) * get_element(A, n, n, i, j);
            }
            for (int i = 0; i < n; ++i) {
                temp[i] -= dot * get_element(Q, n, n, i, k);
            }
        }
        // Normalize
        double norm = 0.0;
        for (int i = 0; i < n; ++i) {
            norm += temp[i] * temp[i];
        }
        norm = std::sqrt(norm);
        if (norm < 1e-10) norm = 1e-10;
        for (int i = 0; i < n; ++i) {
            set_element(Q, n, n, i, j, temp[i] / norm);
        }
    }
    delete[] temp;
    return Q;
}

// Compute condition number from singular values
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

// gallery_randsvd function
double* gallery_randsvd(int n, double kappa = -1.0, int mode = 3, int kl = -1, int ku = -1, int method = 0, int random_state = 42) {
    // Initialize random number generator
    std::mt19937 rng(random_state);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    // Handle matrix dimensions (square only)
    if (n < 1) {
        std::cerr << "n must be a positive integer\n";
        return nullptr;
    }
    int m = n;
    
    // Default kappa
    if (kappa < 0) {
        kappa = std::sqrt(1.0 / std::numeric_limits<double>::epsilon());
    }
    
    // Default kl and ku
    if (kl < 0) kl = m - 1;
    if (ku < 0) ku = kl;
    if (kl >= m || ku >= n || kl < 0 || ku < 0) {
        std::cerr << "kl and ku must be non-negative and less than matrix dimensions\n";
        return nullptr;
    }
    
    // Validate mode and method
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
    
    // Symmetric positive definite case: kappa <= 1
    if (kappa <= 1) {
        if (m != n) {
            std::cerr << "For kappa <= 1, matrix must be square (m == n)\n";
            delete[] sigma;
            return nullptr;
        }
        double lambda_min = std::abs(kappa);
        double lambda_max = 1.0;
        
        // Generate eigenvalues
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
        
        // Sort eigenvalues
        if (mode < 0) {
            std::sort(sigma, sigma + p);
        } else {
            std::sort(sigma, sigma + p, std::greater<double>());
        }
        
        // Check for positive eigenvalues
        for (int i = 0; i < p; ++i) {
            if (sigma[i] <= 0) {
                std::cerr << "Eigenvalues must be positive for symmetric positive definite matrix\n";
                delete[] sigma;
                return nullptr;
            }
        }
        
        // Generate random matrix X
        double* X = allocate_matrix(n, n);
        for (int i = 0; i < n * n; ++i) {
            X[i] = dist(rng);
        }
        
        // QR decomposition to get Q
        double* Q = gram_schmidt(X, n);
        deallocate_matrix(X);
        
        // Create diagonal matrix
        double* D = allocate_matrix(n, n);
        for (int i = 0; i < n; ++i) {
            set_element(D, n, n, i, i, sigma[i]);
        }
        
        // Compute A = Q * D * Q^T
        double* QT = transpose(Q, n, n);
        double* temp = matrix_multiply(D, n, n, QT, n, n);
        double* A = matrix_multiply(Q, n, n, temp, n, n);
        
        // Compute condition number
        std::cout << "Condition number: " << compute_condition_number(sigma, p, kl, ku, n, n) << "\n";
        
        deallocate_matrix(Q);
        deallocate_matrix(D);
        deallocate_matrix(QT);
        deallocate_matrix(temp);
        delete[] sigma;
        return A;
    }
    
    // General case: kappa > 1
    if (std::abs(kappa) < 1) {
        std::cerr << "For non-symmetric case, abs(kappa) must be >= 1\n";
        delete[] sigma;
        return nullptr;
    }
    
    double sigma_max = 1.0;
    double sigma_min = sigma_max / std::abs(kappa);
    
    // Generate singular values
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
    
    // Sort singular values
    if (mode < 0) {
        std::sort(sigma, sigma + p);
    } else {
        std::sort(sigma, sigma + p, std::greater<double>());
    }
    
    // Print singular values
    std::cout << "Generated sigma for mode=" << mode << ": ";
    for (int i = 0; i < p; ++i) {
        std::cout << sigma[i] << " ";
    }
    std::cout << "\n";
    
    // Create Sigma matrix
    double* Sigma = allocate_matrix(m, n);
    for (int i = 0; i < p; ++i) {
        set_element(Sigma, m, n, i, i, sigma[i]);
    }
    
    // Generate U and V
    double* X = allocate_matrix(m, m);
    double* Y = allocate_matrix(n, n);
    for (int i = 0; i < m * m; ++i) {
        X[i] = dist(rng);
    }
    for (int i = 0; i < n * n; ++i) {
        Y[i] = dist(rng);
    }
    
    double* U = gram_schmidt(X, m);
    double* V = gram_schmidt(Y, n);
    deallocate_matrix(X);
    deallocate_matrix(Y);
    
    // Compute A = U * Sigma * V^T
    double* VT = transpose(V, n, n);
    double* temp = matrix_multiply(Sigma, m, n, VT, n, n);
    double* A = matrix_multiply(U, m, m, temp, m, n);
    
    // Apply banded structure
    if (kl < m - 1 || ku < n - 1) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j < i - kl || j > i + ku) {
                    set_element(A, m, n, i, j, 0);
                }
            }
        }
    }
    
    // Compute condition number
    std::cout << "Condition number: " << compute_condition_number(sigma, p, kl, ku, m, n) << "\n";
    
    deallocate_matrix(U);
    deallocate_matrix(V);
    deallocate_matrix(VT);
    deallocate_matrix(Sigma);
    deallocate_matrix(temp);
    delete[] sigma;
    
    return A;
}

// Test function
void test_gallery_randsvd() {
    // Test 1: Square matrix, n=4, kappa=100, mode=3
    std::cout << "\nTest 1: Square matrix, n=4, kappa=256\n";
    double* A = gallery_randsvd(4, 256, 3);
    std::cout << "Matrix A:\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << get_element(A, 4, 4, i, j) << " ";
        }
        std::cout << "\n";
    }
    deallocate_matrix(A);
    
    // Test 2: Square matrix, n=4, kappa=100, mode=1
    std::cout << "\nTest 2: Square matrix, n=4, kappa=100, mode=1\n";
    A = gallery_randsvd(4, 100.0, 1);
    std::cout << "Matrix A:\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << get_element(A, 4, 4, i, j) << " ";
        }
        std::cout << "\n";
    }
    deallocate_matrix(A);
    
    // Test 3: Banded matrix, n=4, kappa=100, mode=3, kl=1, ku=1
    std::cout << "\nTest 3: Banded matrix, n=4, kappa=100, mode=3, kl=1, ku=1\n";
    A = gallery_randsvd(4, 100.0, 3, 1, 1);
    std::cout << "Matrix A:\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << get_element(A, 4, 4, i, j) << " ";
        }
        std::cout << "\n";
    }
    deallocate_matrix(A);
    
    // Test 5: Symmetric positive definite, n=4, kappa=0.5, mode=3
    std::cout << "\nTest 5: Symmetric positive definite, n=4, kappa=0.5, mode=3\n";
    A = gallery_randsvd(4, 0.5, 3);
    std::cout << "Matrix A:\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << get_element(A, 4, 4, i, j) << " ";
        }
        std::cout << "\n";
    }
    deallocate_matrix(A);
}

int main() {
    test_gallery_randsvd();
    return 0;
}