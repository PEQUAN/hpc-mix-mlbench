#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>

struct CSRMatrix {
    int n;           // Matrix dimension
    double* values;  // Non-zero values
    int* col_indices;// Column indices of non-zeros
    int* row_ptr;    // Row pointers
    int nnz;         // Number of non-zero elements
};

struct Pair {
    int first;
    double second;
};

struct SORResult {
    double* x;
    flx::floatx<5, 2> residual;
    int iterations;
    bool converged;
};

bool compare_by_column(const Pair& a, const Pair& b) {
    return a.first < b.first;
}

void free_csr_matrix(CSRMatrix& A) {
    delete[] A.values;
    delete[] A.col_indices;
    delete[] A.row_ptr;
    A.values = nullptr;
    A.col_indices = nullptr;
    A.row_ptr = nullptr;
}

CSRMatrix generate_random_matrix(int n, float sparsity = 0.01) {
    CSRMatrix A = {n, nullptr, nullptr, nullptr, 0};
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    std::uniform_real_distribution<> prob(0.0, 1.0);

    Pair** temp = new Pair*[n]();
    int* temp_sizes = new int[n]();
    for (int i = 0; i < n; ++i) {
        temp[i] = new Pair[n]();
    }

    for (int i = 0; i < n; ++i) {
        temp[i][temp_sizes[i]] = {i, 0.0}; 
        temp_sizes[i]++;
        for (int j = 0; j < n; ++j) {
            if (i != j && prob(gen) < sparsity) {
                temp[i][temp_sizes[i]] = {j, dis(gen)};
                temp_sizes[i]++;
                if (i < j) {
                    temp[j][temp_sizes[j]] = {i, temp[i][temp_sizes[i]-1].second};
                    temp_sizes[j]++;
                }
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        double off_diag_sum = 0.0;
        for (int k = 0; k < temp_sizes[i]; ++k) {
            if (temp[i][k].first != i) {
                off_diag_sum += abs(temp[i][k].second);
            }
        }
        for (int k = 0; k < temp_sizes[i]; ++k) {
            if (temp[i][k].first == i) {
                temp[i][k].second = off_diag_sum + 1.0;
                break;
            }
        }
    }

    A.nnz = 0;
    for (int i = 0; i < n; ++i) {
        A.nnz += temp_sizes[i];
    }

    A.values = new double[A.nnz];
    A.col_indices = new int[A.nnz];
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;

    int pos = 0;
    for (int i = 0; i < n; ++i) {
        std::sort(temp[i], temp[i] + temp_sizes[i], compare_by_column);
        A.row_ptr[i + 1] = A.row_ptr[i] + temp_sizes[i];
        for (int k = 0; k < temp_sizes[i]; ++k) {
            A.col_indices[pos] = temp[i][k].first;
            A.values[pos] = temp[i][k].second;
            pos++;
        }
    }

    for (int i = 0; i < n; ++i) delete[] temp[i];
    delete[] temp;
    delete[] temp_sizes;

    std::cout << "Generated matrix: " << n << " x " << n << " with " << A.nnz << " non-zeros" << std::endl;
    return A;
}

double* matvec(const CSRMatrix& A, const double* x) {
    double* y = new double[A.n]();
    for (int i = 0; i < A.n; ++i) {
        float sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            sum += A.values[j] * x[A.col_indices[j]];
        }
        y[i] = sum;
    }
    return y;
}

flx::floatx<5, 2> norm(const double* v, int n) {
    float d = 0.0;
    for (int i = 0; i < n; ++i) {
        d += v[i] * v[i];
    }
    return sqrt(d);
}

double* axpy(flx::floatx<5, 2> alpha, const flx::floatx<5, 2>* x, const flx::floatx<5, 2>* y, int n) {
    double* result = new double[n];
    for (int i = 0; i < n; ++i) {
        result[i] = alpha*x[i] + y[i];
    }
    return result;
}

double* get_diagonal(const CSRMatrix& A) {
    double* diag = new double[A.n]();
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] == i) {
                diag[i] = A.values[j];
                break;
            }
        }
    }
    return diag;
}


SORResult sor(const CSRMatrix& A, const double* b, float omega, int max_iter = 5000, flx::floatx<8, 7> tol = 1e-6) {
    if (omega <= 0.0 || omega >= 2.0) {
        std::cerr << "Error: Omega must be between 0 and 2" << std::endl;
        double* x = new double[A.n]();
        return {x, 0.0, 0, false};
    }

    int n = A.n;
    double* x = new double[n];
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.1, 1.0);
    for (int i = 0; i < n; ++i) {
        x[i] = dis(gen);
    }
    double* r = new double[n];
    for (int i = 0; i < n; ++i) {
        r[i] = b[i];
    }
    flx::floatx<5, 2> initial_norm = norm(r, n);
    flx::floatx<5, 2> tol_abs = tol * initial_norm;
    if (initial_norm < 1e-10) tol_abs = tol;
    const flx::floatx<5, 2> eps = 2.2204460492503131e-16;

    double* diag = get_diagonal(A);
    double* b_scaled = new double[n];
    for (int i = 0; i < n; ++i) {
        b_scaled[i] = (abs(diag[i]) > eps ? b[i] / diag[i] : b[i]);
    }
    double* values_scaled = new double[A.nnz];
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            values_scaled[j] = (abs(diag[i]) > eps ? A.values[j] / diag[i] : A.values[j]);
        }
    }

    if (initial_norm < eps) {
        delete[] r;
        delete[] diag;
        delete[] b_scaled;
        delete[] values_scaled;
        return {x, initial_norm, 0, true};
    }

    int iter;
    for (iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
                int col = A.col_indices[j];
                if (col != i) {
                    sum += values_scaled[j] * x[col];
                }
            }
            flx::floatx<5, 2> diag_val = abs(diag[i]) > eps ? 1.0 : 0.0;
            if (diag_val < eps) {
                std::cerr << "Error: Zero diagonal element at row " << i << std::endl;
                delete[] x;
                delete[] r;
                delete[] diag;
                delete[] b_scaled;
                delete[] values_scaled;
                return {new double[n](), 0.0, iter, false};
            }
            x[i] = (1.0 - omega) * x[i] + (omega / diag_val) * (b_scaled[i] - sum);
        }

        double* Ax = matvec(A, x);
        for (int i = 0; i < n; ++i) {
            r[i] = b[i] - Ax[i];
        }
        delete[] Ax;
        flx::floatx<5, 2> r_norm = norm(r, n);

        if (r_norm < tol_abs) {
            std::cout << "Converged at iteration " << iter + 1 << std::endl;
            delete[] r;
            delete[] diag;
            delete[] b_scaled;
            delete[] values_scaled;
            return {x, r_norm, iter + 1, true};
        }
    }

    std::cout << "Max iterations reached: " << iter << std::endl;
    delete[] r;
    delete[] diag;
    delete[] b_scaled;
    delete[] values_scaled;
    return {x, norm(r, n), iter, false};
}

double* generate_rhs(int n) {
    double* b = new double[n];
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}


int main() {
    try {
        int n = 1000; // Matrix size
        float sparsity = 0.01; // 1% non-zeros
        CSRMatrix A = generate_random_matrix(n, sparsity);
        if (A.n == 0) {
            free_csr_matrix(A);
            return 1;
        }

        double* b = generate_rhs(A.n);
        double* diag = get_diagonal(A);
        float omega = 1.2;
        std::cout << "Estimated omega: " << omega << std::endl;
        delete[] diag;

        SORResult result = sor(A, b, omega, 5000, 1e-6);

        std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
        std::cout << "Final residual: " << result.residual << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Converged: " << (result.converged ? "yes" : "no") << std::endl;


        double check_x[A.n];
        // add for check
        for (int i=0; i<A.n; i++){
            check_x[i] = result.x[i];
        }


        PROMISE_CHECK_ARRAY(check_x, A.n);
        free_csr_matrix(A);
        delete[] b;
        delete[] result.x;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}