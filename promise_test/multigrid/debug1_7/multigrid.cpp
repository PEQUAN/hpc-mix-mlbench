#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <fstream>

struct Matrix {
    int n;       
    float* data;
};

Matrix create_matrix(int n) {
    Matrix A;
    A.n = n;
    A.data = new float[n * n]();
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
    std::uniform_real_distribution<> dis(0.1, 0.5); 

    int h = static_cast<int>(sqrt((double)n));
    if (h * h != n) {
        std::cerr << "Error: n must be a perfect square for 2D grid" << std::endl;
        free_matrix(A);
        return A;
    }
    for (int i = 0; i < n; ++i) {
        int row = i / h;
        int col = i % h;
        A.data[i * n + i] = 4.0; 
        if (col > 0) { 
            float val = -1.0 + dis(gen);
            A.data[i * n + (i - 1)] = val;
            A.data[(i - 1) * n + i] = val; 
        }
        if (col < h - 1) { 
            float val = -1.0 + dis(gen);
            A.data[i * n + (i + 1)] = val;
            A.data[(i + 1) * n + i] = val;
        }
        if (row > 0) { 
            float val = -1.0 + dis(gen);
            A.data[i * n + (i - h)] = val;
            A.data[(i - h) * n + i] = val;
        }
        if (row < h - 1) { 
            float val = -1.0 + dis(gen);
            A.data[i * n + (i + h)] = val;
            A.data[(i + h) * n + i] = val;
        }
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

float dot(const double* a, const double* b, int n) {
    float result = 0.0;
    for (int i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

float norm(const double* v, int n) {
    return sqrt(dot(v, v, n));
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

void gauss_seidel_smoother(const Matrix& A, double* x, const double* b, int num_iter) {
    int n = A.n;
    for (int iter = 0; iter < num_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            double sigma = 0.0;
            for (int j = 0; j < i; ++j) {
                sigma += A.data[i * n + j] * x[j];
            }
            for (int j = i + 1; j < n; ++j) {
                sigma += A.data[i * n + j] * x[j];
            }
            if (abs(A.data[i * n + i]) < 1e-15) {
                std::cerr << "Zero diagonal element at i=" << i << std::endl;
                return;
            }
            x[i] = (b[i] - sigma) / A.data[i * n + i];
        }
    }
}

double* restrict_vector(const double* r_fine, int n_fine) {
    int h_fine = static_cast<int>(sqrt(n_fine));
    int h_coarse = h_fine / 2;
    int n_coarse = h_coarse * h_coarse;
    double* r_coarse = new double[n_coarse]();

    for (int i = 0; i < h_coarse; ++i) {
        for (int j = 0; j < h_coarse; ++j) {
            int idx_coarse = i * h_coarse + j;
            int idx_fine = (2 * i) * h_fine + (2 * j);
            double sum = r_fine[idx_fine];
            if (j + 1 < h_fine) {
                sum += r_fine[idx_fine + 1];
            } else {
                sum += 0.0;
            }
            if (i + 1 < h_fine) {
                sum += r_fine[idx_fine + h_fine];
            } else {
                sum += 0.0;
            }
            if (i + 1 < h_fine && j + 1 < h_fine) {
                sum += r_fine[idx_fine + h_fine + 1];
            } else {
                sum += 0.0;
            }
            r_coarse[idx_coarse] = 0.25 * sum;
        }
    }
    return r_coarse;
}

double* prolong_vector(const double* e_coarse, int n_coarse) {
    int h_coarse = static_cast<int>(sqrt(n_coarse));
    int h_fine = 2 * h_coarse;
    int n_fine = h_fine * h_fine;
    double* e_fine = new double[n_fine]();

    for (int i = 0; i < h_coarse; ++i) {
        for (int j = 0; j < h_coarse; ++j) {
            int idx_coarse = i * h_coarse + j;
            int idx_fine = (2 * i) * h_fine + (2 * j);
            e_fine[idx_fine] = e_coarse[idx_coarse];
            if (2 * j + 1 < h_fine) {
                e_fine[idx_fine + 1] = 0.5 * e_coarse[idx_coarse];
                if (j + 1 < h_coarse) {
                    e_fine[idx_fine + 1] += 0.5 * e_coarse[idx_coarse + 1];
                }
            }
            if (2 * i + 1 < h_fine) {
                e_fine[idx_fine + h_fine] = 0.5 * e_coarse[idx_coarse];
                if (i + 1 < h_coarse) {
                    e_fine[idx_fine + h_fine] += 0.5 * e_coarse[idx_coarse + h_coarse];
                }
            }
            if (2 * i + 1 < h_fine && 2 * j + 1 < h_fine) {
                e_fine[idx_fine + h_fine + 1] = 0.25 * e_coarse[idx_coarse];
                if (j + 1 < h_coarse) {
                    e_fine[idx_fine + h_fine + 1] += 0.25 * e_coarse[idx_coarse + 1];
                }
                if (i + 1 < h_coarse) {
                    e_fine[idx_fine + h_fine + 1] += 0.25 * e_coarse[idx_coarse + h_coarse];
                }
                if (i + 1 < h_coarse && j + 1 < h_coarse) {
                    e_fine[idx_fine + h_fine + 1] += 0.25 * e_coarse[idx_coarse + h_coarse + 1];
                }
            }
        }
    }
    return e_fine;
}

Matrix restrict_matrix(const Matrix& A_fine) {
    int h_fine = static_cast<int>(sqrt(A_fine.n));
    int h_coarse = h_fine / 2;
    int n_coarse = h_coarse * h_coarse;
    Matrix A_coarse = create_matrix(n_coarse);

    for (int i = 0; i < h_coarse; ++i) {
        for (int j = 0; j < h_coarse; ++j) {
            int idx_coarse = i * h_coarse + j;
            int idx_fine = (2 * i) * h_fine + (2 * j);
            A_coarse.data[idx_coarse * n_coarse + idx_coarse] = 4.0;
            if (j > 0) {
                A_coarse.data[idx_coarse * n_coarse + (idx_coarse - 1)] = -1.0;
                A_coarse.data[(idx_coarse - 1) * n_coarse + idx_coarse] = -1.0;
            }
            if (j < h_coarse - 1) {
                A_coarse.data[idx_coarse * n_coarse + (idx_coarse + 1)] = -1.0;
                A_coarse.data[(idx_coarse + 1) * n_coarse + idx_coarse] = -1.0;
            }
            if (i > 0) {
                A_coarse.data[idx_coarse * n_coarse + (idx_coarse - h_coarse)] = -1.0;
                A_coarse.data[(idx_coarse - h_coarse) * n_coarse + idx_coarse] = -1.0;
            }
            if (i < h_coarse - 1) {
                A_coarse.data[idx_coarse * n_coarse + (idx_coarse + h_coarse)] = -1.0;
                A_coarse.data[(idx_coarse + h_coarse) * n_coarse + idx_coarse] = -1.0;
            }
        }
    }
    return A_coarse;
}

void v_cycle(const Matrix& A, double* x, const double* b, int num_pre = 2, int num_post = 2) {
    int n = A.n;
    int h = static_cast<int>(sqrt((double)n));
    if (h <= 1) {
        gauss_seidel_smoother(A, x, b, 10); 
        return;
    }

    gauss_seidel_smoother(A, x, b, num_pre);

    double* r = new double[n];
    double* Ax = matvec(A, x);
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - Ax[i];
    }
    delete[] Ax;

    double* r_coarse = restrict_vector(r, n);
    int n_coarse = (h / 2) * (h / 2);

    Matrix A_coarse = restrict_matrix(A);
     double* e_coarse = new double[n_coarse]();
    v_cycle(A_coarse, e_coarse, r_coarse, num_pre, num_post);

    double* e_fine = prolong_vector(e_coarse, n_coarse);
    for (int i = 0; i < n; ++i) {
        x[i] += e_fine[i];
    }

    gauss_seidel_smoother(A, x, b, num_post);

    delete[] r;
    delete[] r_coarse;
    delete[] e_coarse;
    delete[] e_fine;
    free_matrix(A_coarse);
}

struct MGResult {
    double* x;
    float residual;
    int iterations;
    bool converged;
};

MGResult multigrid(const Matrix& A, const double* b, int max_iter = 20, float tol = 1e-6) {
    int n = A.n;
    double* x = new double[n]();
    double* r = new double[n];
    for (int i = 0; i < n; ++i) {
        r[i] = b[i];
    }
    float initial_norm = norm(r, n);
    float tol_abs = tol * initial_norm;

    for (int iter = 0; iter < max_iter; ++iter) {
        v_cycle(A, x, b);

        double* Ax = matvec(A, x);
        for (int i = 0; i < n; ++i) {
            r[i] = b[i] - Ax[i];
        }
        delete[] Ax;

        float r_norm = norm(r, n);
        if (r_norm < tol_abs) {
            std::cout << "Converged at iteration " << iter + 1 << std::endl;
            delete[] r;
            return {x, r_norm, iter + 1, true};
        }
    }

    std::cout << "Max iterations reached" << std::endl;
    delete[] r;
    return {x, norm(r, n), max_iter, false};
}


int main() {
    try {
        int h = 32; 
        int n = h * h;
        Matrix A = generate_random_matrix(n);
        if (A.n == 0) {
            return 1;
        }
        double* b = generate_rhs(n);

        MGResult result = multigrid(A, b);

        double* Ax = matvec(A, result.x);
        double* residual = new double[n];
        for (int i = 0; i < n; ++i) {
            residual[i] = b[i] - Ax[i];
        }
        float residual_norm = norm(residual, n);

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