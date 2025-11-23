#include <half.hpp>
#include <floatx.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>

struct CSRMatrix {
    int n;           
    float* values;  
    int* col_indices;
    int* row_ptr;    
    int nnz;        
};

struct Pair {
    int first;
    flx::floatx<5, 10> second;
};

struct SORResult {
    float* x;
    flx::floatx<4, 3> residual;
    int iterations;
    bool converged;
};

bool compare_by_column(Pair& a, Pair& b) {
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

CSRMatrix read_mtx(std::string& filename) {
    CSRMatrix A = {0, nullptr, nullptr, nullptr, 0};
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return A;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] != '%') break;
    }

    std::istringstream header(line);
    int m, n, nnz;
    header >> m >> n >> nnz;
    if (m != n) {
        std::cerr << "Error: Matrix must be square" << std::endl;
        return A;
    }

    Pair* entries = new Pair[nnz];
    int count = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '%') continue;
        std::istringstream iss(line);
        int row, col;
        flx::floatx<5, 10> val;
        if (iss >> row >> col >> val) {
            row--; col--; // Convert to 0-based
            entries[count++] = {row * n + col, val};
        }
    }
    file.close();
    if (count != nnz) {
        std::cerr << "Error: Read " << count << " entries, expected " << nnz << std::endl;
        delete[] entries;
        return A;
    }

    int* row_counts = new int[n]();
    for (int i = 0; i < nnz; ++i) {
        int row = entries[i].first / n;
        row_counts[row]++;
    }

    A.n = n;
    A.nnz = nnz;
    A.values = new float[nnz];
    A.col_indices = new int[nnz];
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;

    int* pos = new int[n]();
    for (int i = 0; i < n; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + row_counts[i];
        pos[i] = A.row_ptr[i];
    }

    for (int i = 0; i < nnz; ++i) {
        int flat = entries[i].first;
        int row = flat / n;
        int col = flat % n;
        int idx = pos[row]++;
        A.values[idx] = entries[i].second;
        A.col_indices[idx] = col;
    }

    delete[] entries;
    delete[] row_counts;
    delete[] pos;

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << A.nnz << " non-zeros from " << filename << std::endl;
    return A;
}

float* matvec(CSRMatrix& A, float* x) {
    float* y = new float[A.n]();
    for (int i = 0; i < A.n; ++i) {
        flx::floatx<5, 10> sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            sum += A.values[j] * x[A.col_indices[j]];
        }
        y[i] = sum;
    }
    return y;
}

flx::floatx<4, 3> norm(float* v, int n) {
    float d = 0.0;
    for (int i = 0; i < n; ++i) {
        d += v[i] * v[i];
    }
    return sqrt(d);
}

float* axpy(flx::floatx<4, 3> alpha, float* x, float* y, int n) {
    float* result = new float[n];
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

float* get_diagonal(CSRMatrix& A) {
    float* diag = new float[A.n]();
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

SORResult sor(CSRMatrix& A, float* b, flx::floatx<4, 3> omega, int max_iter = 5000, flx::floatx<4, 3> tol = 1e-6) {
    if (omega <= 0.0 || omega >= 2.0) {
        std::cerr << "Error: Omega must be between 0 and 2" << std::endl;
        float* x = new float[A.n]();
        return {x, 0.0, 0, false};
    }

    int n = A.n;
    float* x = new float[n];
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.1, 1.0);
    for (int i = 0; i < n; ++i) {
        x[i] = dis(gen);
    }
    float* r = new float[n];
    for (int i = 0; i < n; ++i) {
        r[i] = b[i];
    }
    flx::floatx<4, 3> initial_norm = norm(r, n);
    flx::floatx<4, 3> tol_abs = tol * initial_norm;
    if (initial_norm < 1e-10) tol_abs = tol;
    flx::floatx<4, 3> eps = std::numeric_limits<flx::floatx<4, 3>>::epsilon();

    float* diag = get_diagonal(A);
    flx::floatx<5, 10>* b_scaled = new flx::floatx<5, 10>[n];
    for (int i = 0; i < n; ++i) {
        b_scaled[i] = (abs(diag[i]) > eps ? b[i] / diag[i] : b[i]);
    }
    
    flx::floatx<4, 3>* values_scaled = new flx::floatx<4, 3>[A.nnz];
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
            flx::floatx<5, 10> sum = 0.0;
            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
                int col = A.col_indices[j];
                if (col != i) {
                    sum += values_scaled[j] * x[col];
                }
            }
            flx::floatx<4, 3> diag_val = abs(diag[i]) > eps ? 1.0 : 0.0;
            if (diag_val < eps) {
                std::cerr << "Error: Zero diagonal element at row " << i << std::endl;
                delete[] x;
                delete[] r;
                delete[] diag;
                delete[] b_scaled;
                delete[] values_scaled;
                return {new float[n](), 0.0, iter, false};
            }
            x[i] = (1.0 - omega) * x[i] + (omega / diag_val) * (b_scaled[i] - sum);
        }

        float* Ax = matvec(A, x);
        for (int i = 0; i < n; ++i) {
            r[i] = b[i] - Ax[i];
        }
        delete[] Ax;
        flx::floatx<4, 3> r_norm = norm(r, n);

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

int main() {
    try {
        std::string mtx_file = "1138_bus.mtx";  // Adjust path if needed
        CSRMatrix A = read_mtx(mtx_file);
        if (A.n == 0 || A.values == nullptr) {
            free_csr_matrix(A);
            return 1;
        }

        int n = A.n;
        float* x_true = new float[n];
        for (int i = 0; i < n; ++i) {
            x_true[i] = 1.0; // Ground truth: x = ones(n)
        }

        // Compute b = A @ x_true
        float* b = matvec(A, x_true);

        float* diag = get_diagonal(A);
        flx::floatx<4, 3> omega = 1.0;
        std::cout << "Using omega: " << omega << std::endl;

        auto start = std::chrono::high_resolution_clock::now();
        SORResult result = sor(A, b, omega, 5000, 1e-6);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Matrix: 1138_bus.mtx (" << A.n << " x " << A.n << ")" << std::endl;
        std::cout << "Time: " << duration.count() << " ms" << std::endl;
        std::cout << "Final residual: " << result.residual << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Converged: " << (result.converged ? "yes" : "no") << std::endl;

        float* error_vec = axpy(-1.0, result.x, x_true, n);
        flx::floatx<4, 3> error = norm(error_vec, n);
        std::cout << "Error ||x - x_true||_2: " << error << std::endl;
        
        double* check_x = new double[n];
        
        for (int i = 0; i < n; ++i) {
            check_x[i] = result.x[i];
        }
        
        PROMISE_CHECK_ARRAY(check_x, n);


        free_csr_matrix(A);
        delete[] b;
        delete[] x_true;
        delete[] result.x;
        delete[] diag;
        delete[] error_vec;
    }
    catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}