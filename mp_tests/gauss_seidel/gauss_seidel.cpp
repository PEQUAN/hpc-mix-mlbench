#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>

struct Matrix {
    int n;
    __PROMISE__* data;
};


Matrix create_matrix(int n) {
    Matrix A;
    A.n = n;
    A.data = new __PROMISE__[n * n]();
    return A;
}


void free_matrix(Matrix& A) {
    delete[] A.data;
    A.data = nullptr;
    A.n = 0;
}


Matrix generate_random_matrix(int n, unsigned int seed = 42) {
    Matrix A = create_matrix(n);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(1.0, 10.0);

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            __PROMISE__ val = dis(gen);
            A.data[i * n + j] = val;
            A.data[j * n + i] = val;
        }
    }

    for (int i = 0; i < n; ++i) {
        A.data[i * n + i] += n;
    }
    return A;
}


__PROMISE__* generate_rhs(int n, unsigned int seed = 42) {
    __PROMISE__* b = new __PROMISE__[n];
    std::mt19937 gen(seed + 1); 
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}


__PROMISE__ dot(const __PROMISE__* a, const __PROMISE__* b, int n) {
    __PROMISE__ result = 0.0;
    for (int i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}


__PROMISE__ norm(const __PROMISE__* v, int n) {
    return sqrt(dot(v, v, n));
}

__PROMISE__* matvec(const Matrix& A, const __PROMISE__* x) {
    __PROMISE__* y = new __PROMISE__[A.n]();
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.n; ++j) {
            y[i] += A.data[i * A.n + j] * x[j];
        }
    }
    return y;
}

struct GSResult {
    __PROMISE__* x;
    __PROMISE__ residual;
    int iterations;
    bool converged;
};



GSResult gauss_seidel(const Matrix& A, const __PROMISE__* b, int max_iter = 1000, __PROMISE__ tol = 1e-6) {
    int n = A.n;
    __PROMISE__* x = new __PROMISE__[n]();
    __PROMISE__* r = new __PROMISE__[n];
    for (int i = 0; i < n; ++i) {
        r[i] = b[i];
    }
    __PROMISE__ initial_norm = norm(r, n);
    __PROMISE__ tol_abs = tol * initial_norm;
    int iter;

    for (iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            __PROMISE__ sigma = 0.0;
            for (int j = 0; j < i; ++j) {
                sigma += A.data[i * n + j] * x[j];
            }
            for (int j = i + 1; j < n; ++j) {
                sigma += A.data[i * n + j] * x[j];
            }
            if (abs(A.data[i * n + i]) < 1e-15) {
                std::cerr << "Zero diagonal element at i=" << i << std::endl;
                delete[] r;
                return {x, norm(r, n), iter, false};
            }
            x[i] = (b[i] - sigma) / A.data[i * n + i];
        }


        __PROMISE__* Ax = matvec(A, x);
        for (int i = 0; i < n; ++i) {
            r[i] = b[i] - Ax[i];
        }
        delete[] Ax;

        __PROMISE__ r_norm = norm(r, n);
        if (r_norm < tol_abs) {
            std::cout << "Converged at iteration " << iter + 1 << std::endl;
            delete[] r;
            return {x, r_norm, iter + 1, true};
        }
    }

    std::cout << "Max iterations reached: " << iter << std::endl;
    delete[] r;
    return {x, norm(r, n), iter, false};
}


int main() {
    try {
        int n = 1000; 
        Matrix A = generate_random_matrix(n);
        __PROMISE__* b = generate_rhs(n);

        auto start = std::chrono::high_resolution_clock::now();
        GSResult result = gauss_seidel(A, b, n);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        
        __PROMISE__* Ax = matvec(A, result.x);
        __PROMISE__* residual = new __PROMISE__[n];
        for (int i = 0; i < n; ++i) {
            residual[i] = b[i] - Ax[i];
        }
        __PROMISE__ residual_norm = norm(residual, n);

        std::cout << "Matrix size: " << n << " x " << n << std::endl;
        std::cout << "Final residual norm: " << residual_norm << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Converged: " << (result.converged ? "yes" : "no") << std::endl;
        double check_x[A.n];
        // add for check
        for (int i=0; i<A.n; i++){
            check_x[i] = result.x[i];
        }



        PROMISE_CHECK_ARRAY(check_x, A.n);
        free_matrix(A);
        delete[] b;
        delete[] result.x;
        delete[] Ax;
        delete[] residual;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}