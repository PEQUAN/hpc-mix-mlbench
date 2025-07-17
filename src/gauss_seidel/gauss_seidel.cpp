#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>

struct Matrix {
    int n;
    double* data;
};


Matrix create_matrix(int n) {
    Matrix A;
    A.n = n;
    A.data = new double[n * n]();
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
            double val = dis(gen);
            A.data[i * n + j] = val;
            A.data[j * n + i] = val;
        }
    }

    for (int i = 0; i < n; ++i) {
        A.data[i * n + i] += n;
    }
    return A;
}


double* generate_rhs(int n, unsigned int seed = 42) {
    double* b = new double[n];
    std::mt19937 gen(seed + 1); 
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}


double dot(const double* a, const double* b, int n) {
    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}


double norm(const double* v, int n) {
    return std::sqrt(dot(v, v, n));
}

double* matvec(const Matrix& A, const double* x) {
    double* y = new double[A.n]();
    for (int i = 0; i < A.n; ++i) {
        for (int j = 0; j < A.n; ++j) {
            y[i] += A.data[i * A.n + j] * x[j];
        }
    }
    return y;
}

struct GSResult {
    double* x;
    double residual;
    int iterations;
    bool converged;
};



GSResult gauss_seidel(const Matrix& A, const double* b, int max_iter = 1000, double tol = 1e-6) {
    int n = A.n;
    double* x = new double[n]();
    double* r = new double[n];
    for (int i = 0; i < n; ++i) {
        r[i] = b[i];
    }
    double initial_norm = norm(r, n);
    double tol_abs = tol * initial_norm;
    int iter;

    for (iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            double sigma = 0.0;
            for (int j = 0; j < i; ++j) {
                sigma += A.data[i * n + j] * x[j];
            }
            for (int j = i + 1; j < n; ++j) {
                sigma += A.data[i * n + j] * x[j];
            }
            if (std::abs(A.data[i * n + i]) < 1e-15) {
                std::cerr << "Zero diagonal element at i=" << i << std::endl;
                delete[] r;
                return {x, norm(r, n), iter, false};
            }
            x[i] = (b[i] - sigma) / A.data[i * n + i];
        }


        double* Ax = matvec(A, x);
        for (int i = 0; i < n; ++i) {
            r[i] = b[i] - Ax[i];
        }
        delete[] Ax;

        double r_norm = norm(r, n);
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


void write_solution(const double* x, int n, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }
    file << "x\n";
    for (int i = 0; i < n; ++i) {
        file << x[i] << "\n";
    }
    file.close();
}

int main() {
    try {
        int n = 1000; 
        Matrix A = generate_random_matrix(n);
        double* b = generate_rhs(n);

        auto start = std::chrono::high_resolution_clock::now();
        GSResult result = gauss_seidel(A, b, n);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        
        double* Ax = matvec(A, result.x);
        double* residual = new double[n];
        for (int i = 0; i < n; ++i) {
            residual[i] = b[i] - Ax[i];
        }
        double residual_norm = norm(residual, n);

        std::cout << "Matrix size: " << n << " x " << n << std::endl;
        std::cout << "Time: " << duration.count() << " ms" << std::endl;
        std::cout << "Final residual norm: " << residual_norm << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Converged: " << (result.converged ? "yes" : "no") << std::endl;

        write_solution(result.x, n, "../results/gauss_seidel/gauss_seidel_solution.csv");

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