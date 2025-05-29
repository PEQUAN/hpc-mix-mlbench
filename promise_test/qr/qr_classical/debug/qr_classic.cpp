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

// Function to get element in row-major order
double get_element(double* matrix, int m, int n, int i, int j) {
    return matrix[i * n + j];
}

// Function to set element in row-major order
void set_element(double* matrix, int m, int n, int i, int j, double value) {
    matrix[i * n + j] = value;
}

// Function to initialize random matrix with fixed seed
void init_random_matrix(double* A, int m, int n, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            set_element(A, m, n, i, j, (double)rand() / RAND_MAX * 10.0); // Random values between 0 and 10
        }
    }
}

// Function to copy matrix
void copy_matrix(double* src, double* dst, int m, int n) {
    for (int i = 0; i < m * n; i++) {
        dst[i] = src[i];
    }
}

// Function to compute dot product of two vectors
float dot_product(double* v1, double* v2, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

// Function to compute vector norm
float vector_norm(double* v, int n) {
    return sqrt(dot_product(v, v, n));
}

// Classical Gram-Schmidt QR decomposition
void classical_gram_schmidt(double* A, double* Q, double* R, int m, int n) {
    double* v = new double[m];
    for (int j = 0; j < n; j++) {
        // Copy column j of A to v
        for (int i = 0; i < m; i++) {
            v[i] = get_element(A, m, n, i, j);
        }
        // Orthogonalize
        for (int k = 0; k < j; k++) {
            float rkj = 0.0;
            for (int i = 0; i < m; i++) {
                rkj += get_element(Q, m, m, i, k) * get_element(A, m, n, i, j);
            }
            set_element(R, m, n, k, j, rkj);
            for (int i = 0; i < m; i++) {
                v[i] -= rkj * get_element(Q, m, m, i, k);
            }
        }
        // Normalize
        float norm = vector_norm(v, m);
        set_element(R, m, n, j, j, norm);
        for (int i = 0; i < m; i++) {
            set_element(Q, m, m, i, j, v[i] / norm);
        }
    }
    delete[] v;
}

float compute_error_A_QR(double* A, double* Q, double* R, int m, int n) {
    double* temp = new double[m * n]; // Compute Frobenius norm of A - QR
    // Compute QR
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += get_element(Q, m, m, i, k) * get_element(R, m, n, k, j);
            }
            temp[i * n + j] = get_element(A, m, n, i, j) - sum;
        }
    }
    // Compute Frobenius norm
    float norm = 0.0;
    for (int i = 0; i < m * n; i++) {
        norm += temp[i] * temp[i];
    }
    delete[] temp;
    return sqrt(norm);
}


float compute_error_QTQ_I(double* Q, int m) {
    half_float::half* temp = new half_float::half[m * m];// Compute Frobenius norm of Q^T Q - I
    // Compute Q^T Q
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            float sum = 0.0;
            for (int k = 0; k < m; k++) {
                sum += get_element(Q, m, m, k, i) * get_element(Q, m, m, k, j);
            }
            temp[i * m + j] = sum - (i == j ? 1.0 : 0.0);
        }
    }
    // Compute Frobenius norm
    float norm = 0.0;
    for (int i = 0; i < m * m; i++) {
        norm += temp[i] * temp[i];
    }
    delete[] temp;
    return sqrt(norm);
}

int main() {
    int m = 100; // Larger matrix size
    int n = 100;

    // Allocate matrices
    double* A = allocate_matrix(m, n);
    double* Q = allocate_matrix(m, m);
    double* R = allocate_matrix(m, n);
    double* A_copy = allocate_matrix(m, n);

    // Initialize random matrix
    init_random_matrix(A, m, n, 32);
    copy_matrix(A, A_copy, m, n);
    printf("Original Matrix A (size %dx%d):\n", m, n);

    // Classical Gram-Schmidt
    classical_gram_schmidt(A, Q, R, m, n);
    printf("Classical Gram-Schmidt:\n");
    printf("Error ||A - QR||: %e\n", compute_error_A_QR(A_copy, Q, R, m, n));
    printf("Error ||Q^T Q - I||: %e\n\n", compute_error_QTQ_I(Q, m));


    double* temp = new double[m * n]; // Compute Frobenius norm of A - QR
    // Compute QR
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += get_element(Q, m, m, i, k) * get_element(R, m, n, k, j);
            }
            temp[i * n + j] = sum;
        }
    }

    PROMISE_CHECK_ARRAY(temp, m*n);
    // Deallocate matrices
    deallocate_matrix(A);
    deallocate_matrix(Q);
    deallocate_matrix(R);
    deallocate_matrix(A_copy);

    return 0;
}