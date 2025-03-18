#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>

// Matrix multiplication: C = A * B
// A is M x K, B is K x N, C is M x N
void matrix_multiply_neon(float* A, float* B, float* C, int M, int N, int K) {
    // Assume M, N, K are multiples of 4 for simplicity
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j += 4) { // Process 4 columns of B at a time
            // Initialize a vector for 4 elements of C[i][j:j+3]
            float32x4_t sum = vdupq_n_f32(0.0f); // Set to zero

            // Inner loop over K
            for (int k = 0; k < K; k++) {
                // Load A[i][k] as a scalar and broadcast it to a vector
                float32x4_t a = vdupq_n_f32(A[i * K + k]);

                // Load 4 elements of B[k][j:j+3] into a vector
                float32x4_t b = vld1q_f32(&B[k * N + j]);

                // Perform fused multiply-add: sum += a * b
                sum = vfmaq_f32(sum, a, b);
            }

            // Store the result back to C[i][j:j+3]
            vst1q_f32(&C[i * N + j], sum);
        }
    }
}

// Naive (non-NEON) matrix multiplication for comparison
void matrix_multiply_naive(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int M = 4, N = 4, K = 4; // Small matrices for simplicity
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices A and B with simple values
    for (int i = 0; i < M * K; i++) A[i] = i + 1;
    for (int i = 0; i < K * N; i++) B[i] = i + 1;

    // Run NEON-optimized matrix multiplication
    matrix_multiply_neon(A, B, C, M, N, K);

    // Print the result
    printf("Result Matrix C (NEON):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.2f ", C[i * N + j]);
        }
        printf("\n");
    }

    // Free memory
    free(A);
    free(B);
    free(C);
    return 0;
}