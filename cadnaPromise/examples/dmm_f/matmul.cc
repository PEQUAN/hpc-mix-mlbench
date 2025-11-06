#include <iostream>
#include <random>
#include <chrono>
#include <cmath>

void matmul(const __PROMISE__* A,  const __PROMISE__* B, __PROMISE__* C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            __PR_scalar__ sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

__PROMISE__ compute_error(const __PROMISE__* C,  const __PROMISE__* C_ref, int n) {
    __PROMISE__ error = 0.0;
    for (int i = 0; i < n * n; ++i) {
        __PROMISE__ diff = C[i] - C_ref[i];
        error += diff * diff;
    }
    return sqrt(error);
}

int main() {
    const int n = 300;
    __PROMISE__* A = new __PROMISE__[n * n];
    __PROMISE__* B = new __PROMISE__[n * n];
    __PROMISE__* C = new __PROMISE__[n * n];
    __PROMISE__* C_ref = new __PROMISE__[n * n];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < n * n; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
        C[i] = 0.0;
        C_ref[i] = 0.0;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += static_cast<double>(A[i * n + k]) * static_cast<double>(B[k * n + j]);
            }
            C_ref[i * n + j] = sum;
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    matmul(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    __PROMISE__ error = compute_error(C, C_ref, n);
    std::cout << "Matrix size: " << n << " x " << n << std::endl;
    std::cout << "Computation time: " << duration.count() << " ms" << std::endl;
    std::cout << "Error: " << error << std::endl;

    PROMISE_CHECK_VAR(error);

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] C_ref;

    return 0;
}