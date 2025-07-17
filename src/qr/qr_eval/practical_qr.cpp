#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

double* allocate_matrix(int m, int n) {
    return new double[m * n];
}

void deallocate_matrix(double* matrix) {
    delete[] matrix;
}

double get_element(double* matrix, int m, int n, int i, int j) {
    return matrix[i * n + j];
}

void set_element(double* matrix, int m, int n, int i, int j, double value) {
    matrix[i * n + j] = value;
}

void init_random_matrix(double* A, int m, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            set_element(A, m, n, i, j, (double)rand() / RAND_MAX * 10.0);
        }
    }
}

void copy_matrix(double* src, double* dst, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        dst[i] = src[i];
    }
}

void matrix_multiply(double* A, double* B, double* C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += get_element(A, m, n, i, k) * get_element(B, n, p, k, j);
            }
            set_element(C, m, p, i, j, sum);
        }
    }
}

double dot_product(double* v1, double* v2, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

double vector_norm(double* v, int n) {
    return sqrt(dot_product(v, v, n));
}

void householder(double* A, double* Q, double* R, int m, int n) {
    // Initialize Q as identity
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            set_element(Q, m, m, i, j, (i == j) ? 1.0 : 0.0);
        }
    }

    copy_matrix(A, R, m, n);
    double* v = new double[m];
    double* w = new double[m]; 

    int k_max = (m < n) ? m : n;
    for (int j = 0; j < k_max; j++) {
        // Reset v
        for (int i = 0; i < m; i++) v[i] = 0.0;

        // Compute Householder vector for column j
        double sigma = 0.0;
        for (int i = j; i < m; i++) {
            v[i] = get_element(R, m, n, i, j);
            sigma += v[i] * v[i];
        }

        double norm = sqrt(sigma);
        if (norm < 1e-14) continue;

        v[j] += (v[j] >= 0.0) ? norm : -norm;
        double v_norm_sq = dot_product(v + j, v + j, m - j);

        if (v_norm_sq < 1e-14) continue;

        // Normalize v
        double v_norm = sqrt(v_norm_sq);
        for (int i = j; i < m; i++) v[i] /= v_norm;

        // Apply Householder to R (from the left)
        for (int k = j; k < n; k++) {
            double dot = 0.0;
            for (int i = j; i < m; i++) {
                dot += v[i] * get_element(R, m, n, i, k);
            }
            for (int i = j; i < m; i++) {
                double val = get_element(R, m, n, i, k) - 2 * dot * v[i];
                set_element(R, m, n, i, k, val);
            }
        }

        // Apply Householder to Q from the left: Q = H * Q
        for (int k = 0; k < m; k++) {
            double dot = 0.0;
            for (int i = j; i < m; i++) {
                dot += v[i] * get_element(Q, m, m, i, k);
            }
            for (int i = j; i < m; i++) {
                double val = get_element(Q, m, m, i, k) - 2 * v[i] * dot;
                set_element(Q, m, m, i, k, val);
            }
        }
    }

    // Transpose Q (since we applied transformations from the right)
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < m; j++) {
            double tmp = get_element(Q, m, m, i, j);
            set_element(Q, m, m, i, j, get_element(Q, m, m, j, i));
            set_element(Q, m, m, j, i, tmp);
        }
    }

    delete[] v;
    delete[] w;
}


double compute_error_A_QHQ(double* A, double* Q, double* H, int n) {
    double* temp = allocate_matrix(n, n);
    double* QT = allocate_matrix(n, n);
    double* HQT = allocate_matrix(n, n);
    
    // Compute Q^T
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            set_element(QT, n, n, i, j, get_element(Q, n, n, j, i));
        }
    }
    
    // Compute H Q^T
    matrix_multiply(H, QT, HQT, n, n, n);
    // Compute Q (H Q^T) = Q H Q^T
    matrix_multiply(Q, HQT, temp, n, n, n);
    // Subtract A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double val = get_element(A, n, n, i, j) - get_element(temp, n, n, i, j);
            set_element(temp, n, n, i, j, val);
        }
    }
    // Compute Frobenius norm
    double norm = 0.0;
    for (int i = 0; i < n * n; i++) {
        norm += temp[i] * temp[i];
    }
    
    deallocate_matrix(temp);
    deallocate_matrix(QT);
    deallocate_matrix(HQT);
    return sqrt(norm);
}

double compute_error_QTQ_I(double* Q, int n) {
    double* temp = allocate_matrix(n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += get_element(Q, n, n, k, i) * get_element(Q, n, n, k, j);
            }
            set_element(temp, n, n, i, j, sum - (i == j ? 1.0 : 0.0));
        }
    }
    double norm = 0.0;
    for (int i = 0; i < n * n; i++) {
        norm += temp[i] * temp[i];
    }
    deallocate_matrix(temp);
    return sqrt(norm);
}

double compute_off_diagonal_norm(double* H, int n) {
    double norm = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                double val = get_element(H, n, n, i, j);
                norm += val * val;
            }
        }
    }
    return sqrt(norm);
}

bool is_converged(double* H, int n, double tol) {
    // Check for upper triangular structure with 2x2 blocks
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Subdiagonal elements (except i-1 for 2x2 blocks)
            if (i > j && (j != i-1 || i == 0)) {
                if (fabs(get_element(H, n, n, i, j)) > tol) {
                    return false;
                }
            }
        }
    }
    return true;
}

double compute_wilkinson_shift(double* A, int n) {
    // Compute shift based on bottom-right 2x2 submatrix
    int i = n - 1;
    double a = get_element(A, n, n, i-1, i-1);
    double b = get_element(A, n, n, i-1, i);
    double c = get_element(A, n, n, i, i-1);
    double d = get_element(A, n, n, i, i);
    
    // Characteristic polynomial: lambda^2 - (a+d)lambda + (ad - bc)
    double trace = a + d;
    double det = a * d - b * c;
    // Eigenvalues: (trace ± sqrt(trace^2 - 4*det)) / 2
    double disc = trace * trace - 4 * det;
    double lambda1 = (trace + sqrt(fabs(disc))) / 2.0;
    double lambda2 = (trace - sqrt(fabs(disc))) / 2.0;
    
    // Choose eigenvalue closer to A(n,n)
    double shift = (fabs(lambda1 - d) < fabs(lambda2 - d)) ? lambda1 : lambda2;
    return shift;
}

void qr_algorithm(double* A, double* Q_total, double* H, double* eigenvalues, int n, int max_iterations) {
    double* Q = allocate_matrix(n, n);
    double* R = allocate_matrix(n, n);
    double* A_next = allocate_matrix(n, n);
    double* temp = allocate_matrix(n, n);
    
    // Initialize Q_total as identity
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            set_element(Q_total, n, n, i, j, (i == j) ? 1.0 : 0.0);
        }
    }
    
    copy_matrix(A, A_next, n, n);
    
    for (int iter = 0; iter < max_iterations; iter++) {
        // Compute Wilkinson shift
        double shift = compute_wilkinson_shift(A_next, n);
        
        // Apply shift: A - μI
        for (int i = 0; i < n; i++) {
            double val = get_element(A_next, n, n, i, i) - shift;
            set_element(A_next, n, n, i, i, val);
        }
        
        // QR factorization using Householder
        householder(A_next, Q, R, n, n);
        
        // Compute A_next = R Q
        matrix_multiply(R, Q, A_next, n, n, n);
        
        // Add shift back: A + μI
        for (int i = 0; i < n; i++) {
            double val = get_element(A_next, n, n, i, i) + shift;
            set_element(A_next, n, n, i, i, val);
        }
        
        // Update Q_total
        copy_matrix(Q_total, temp, n, n);
        matrix_multiply(temp, Q, Q_total, n, n, n);
        
        // Check convergence
        if (is_converged(A_next, n, 1e-10)) {
            printf("Converged after %d iterations.\n", iter + 1);
            break;
        }
    }
    
    copy_matrix(A_next, H, n, n);
    
    // Extract eigenvalues (handle 2x2 blocks)
    for (int i = 0; i < n; i++) {
        if (i < n-1 && fabs(get_element(H, n, n, i+1, i)) > 1e-10) {
            // 2x2 block
            double a = get_element(H, n, n, i, i);
            double b = get_element(H, n, n, i, i+1);
            double c = get_element(H, n, n, i+1, i);
            double d = get_element(H, n, n, i+1, i+1);
            double trace = a + d;
            double det = a * d - b * c;
            // Store real part (simplified)
            eigenvalues[i] = trace / 2.0;
            eigenvalues[i+1] = trace / 2.0; // Complex pair
            i++; // Skip next index
        } else {
            eigenvalues[i] = get_element(H, n, n, i, i);
        }
    }
    
    deallocate_matrix(Q);
    deallocate_matrix(R);
    deallocate_matrix(A_next);
    deallocate_matrix(temp);
}

void print_matrix(double* A, int m, int n, const char* name) {
    printf("%s (size %dx%d):\n", name, m, n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8.4f ", get_element(A, m, n, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

void print_eigenpairs(double* eigenvalues, double* eigenvectors, int n) {
    printf("Eigenvalues (real parts):\n");
    for (int i = 0; i < n; i++) {
        printf("%8.4f ", eigenvalues[i]);
    }
    printf("\n\nEigenvectors (columns):\n");
    print_matrix(eigenvectors, n, n, "V");
}

int main() {
    int n = 20;
    int max_iterations = 1000;
    
    double* A = allocate_matrix(n, n);
    double* Q_total = allocate_matrix(n, n);
    double* H = allocate_matrix(n, n);
    double* eigenvalues = new double[n];
    double* A_copy = allocate_matrix(n, n);
    
    init_random_matrix(A, n, n, 32);
    copy_matrix(A, A_copy, n, n);
    printf("Original Matrix A (size %dx%d, non-symmetric):\n", n, n);
    // print_matrix(A, n, n, "A");
    
    qr_algorithm(A, Q_total, H, eigenvalues, n, max_iterations);
    
    printf("Results for Householder QR method:\n");
    // print_eigenpairs(eigenvalues, Q_total, n);
    // print_matrix(H, n, n, "Final H (triangular)");
    
    double error_A_QHQ = compute_error_A_QHQ(A_copy, Q_total, H, n);
    double error_QTQ_I = compute_error_QTQ_I(Q_total, n);
    double off_diagonal_norm = compute_off_diagonal_norm(H, n);
    printf("Reconstruction Error ||A - Q H Q^T||_F: %e\n", error_A_QHQ);
    printf("Orthogonality Error ||Q^T Q - I||_F: %e\n", error_QTQ_I);
    printf("Off-Diagonal Norm of H: %e\n", off_diagonal_norm);
    
    deallocate_matrix(A);
    deallocate_matrix(Q_total);
    deallocate_matrix(H);
    deallocate_matrix(A_copy);
    delete[] eigenvalues;
    
    return 0;
}