#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
            set_element(A, m, n, i, j, (double)rand() / RAND_MAX * 10.0); // Random values between 0 and 10
        }
    }
}

void copy_matrix(double* src, double* dst, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        dst[i] = src[i];
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

void classical_gram_schmidt(double* A, double* Q, double* R, int m, int n) {
    double* v = new double[m];
    for (int j = 0; j < n; j++) {
        // Copy column j of A to v
        for (int i = 0; i < m; i++) {
            v[i] = get_element(A, m, n, i, j);
        }
        // Orthogonalize
        for (int k = 0; k < j; k++) {
            double rkj = 0.0;
            for (int i = 0; i < m; i++) {
                rkj += get_element(Q, m, m, i, k) * get_element(A, m, n, i, j);
            }
            set_element(R, m, n, k, j, rkj);
            for (int i = 0; i < m; i++) {
                v[i] -= rkj * get_element(Q, m, m, i, k);
            }
        }
        // Normalize
        double norm = vector_norm(v, m);
        set_element(R, m, n, j, j, norm);
        for (int i = 0; i < m; i++) {
            set_element(Q, m, m, i, j, v[i] / norm);
        }
    }
    delete[] v;
}

void modified_gram_schmidt(double* A, double* Q, double* R, int m, int n) {
    double* v = new double[m];
    for (int j = 0; j < n; j++) {
        // Copy column j of A to v
        for (int i = 0; i < m; i++) {
            v[i] = get_element(A, m, n, i, j);
        }
        // Orthogonalize
        for (int k = 0; k < j; k++) {
            double rkj = 0.0;
            for (int i = 0; i < m; i++) {
                rkj += get_element(Q, m, m, i, k) * v[i];
            }
            set_element(R, m, n, k, j, rkj);
            for (int i = 0; i < m; i++) {
                v[i] -= rkj * get_element(Q, m, m, i, k);
            }
        }
        // Normalize
        double norm = vector_norm(v, m);
        set_element(R, m, n, j, j, norm);
        for (int i = 0; i < m; i++) {
            set_element(Q, m, m, i, j, v[i] / norm);
        }
    }
    delete[] v;
}

void householder(double* A, double* Q, double* R, int m, int n) {
    // Initialize Q as identity
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            set_element(Q, m, m, i, j, (i == j) ? 1.0 : 0.0);
        }
    }

    // Copy A to R
    copy_matrix(A, R, m, n);
    double* v = new double[m];
    double* w = new double[m]; // workspace

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



double compute_error_A_QR(double* A, double* Q, double* R, int m, int n) {
    double* temp = new double[m * n]; // Compute Frobenius norm of A - QR
    // Compute QR
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += get_element(Q, m, m, i, k) * get_element(R, m, n, k, j);
            }
            temp[i * n + j] = get_element(A, m, n, i, j) - sum;
        }
    }
    // Compute Frobenius norm
    double norm = 0.0;
    for (int i = 0; i < m * n; i++) {
        norm += temp[i] * temp[i];
    }
    delete[] temp;
    return sqrt(norm);
}


double compute_error_QTQ_I(double* Q, int m) {
    double* temp = new double[m * m];// Compute Frobenius norm of Q^T Q - I
    // Compute Q^T Q
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += get_element(Q, m, m, k, i) * get_element(Q, m, m, k, j);
            }
            temp[i * m + j] = sum - (i == j ? 1.0 : 0.0);
        }
    }
    // Compute Frobenius norm
    double norm = 0.0;
    for (int i = 0; i < m * m; i++) {
        norm += temp[i] * temp[i];
    }
    delete[] temp;
    return sqrt(norm);
}

void print_matrix(double* A, int m, int n, const char* name) {
    printf("%s (showing first 5x5):\n", name);
    int rows = (m < 5) ? m : 5;
    int cols = (n < 5) ? n : 5;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%8.4f ", get_element(A, m, n, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    int m = 100;
    int n = 100;

    // Allocate matrices
    double* A = allocate_matrix(m, n);
    double* Q = allocate_matrix(m, m);
    double* R = allocate_matrix(m, n);
    double* A_copy = allocate_matrix(m, n);

    init_random_matrix(A, m, n, 32);
    copy_matrix(A, A_copy, m, n);
    printf("Original Matrix A (size %dx%d):\n", m, n);
    print_matrix(A, m, n, "A");

    classical_gram_schmidt(A, Q, R, m, n);
    printf("Classical Gram-Schmidt:\n");
    print_matrix(Q, m, m, "Q");
    print_matrix(R, m, n, "R");
    printf("Error ||A - QR||: %e\n", compute_error_A_QR(A_copy, Q, R, m, n));
    printf("Error ||Q^T Q - I||: %e\n\n", compute_error_QTQ_I(Q, m));

    copy_matrix(A_copy, A, m, n);
    modified_gram_schmidt(A, Q, R, m, n);
    printf("Modified Gram-Schmidt:\n");
    print_matrix(Q, m, m, "Q");
    print_matrix(R, m, n, "R");
    printf("Error ||A - QR||: %e\n", compute_error_A_QR(A_copy, Q, R, m, n));
    printf("Error ||Q^T Q - I||: %e\n\n", compute_error_QTQ_I(Q, m));

    copy_matrix(A_copy, A, m, n);
    householder(A, Q, R, m, n);
    printf("Householder:\n");
    print_matrix(Q, m, m, "Q");
    print_matrix(R, m, n, "R");
    printf("Error ||A - QR||: %e\n", compute_error_A_QR(A_copy, Q, R, m, n));
    printf("Error ||Q^T Q - I||: %e\n\n", compute_error_QTQ_I(Q, m));

    deallocate_matrix(A);
    deallocate_matrix(Q);
    deallocate_matrix(R);
    deallocate_matrix(A_copy);

    return 0;
}