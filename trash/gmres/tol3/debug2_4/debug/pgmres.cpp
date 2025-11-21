#include <half.hpp>
#include <floatx.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <vector> // Added for diagonal preconditioner

struct CSRMatrix {
    int n;
    double* values;
    int* col_indices;
    int* row_ptr;
    int nnz;
};

struct Entry { int row, col; flx::floatx<5, 2> val; };

struct Result {
    double* x;
    float residual;
    int iterations;
    double* residual_history;
    int residual_history_size;
};

CSRMatrix read_mtx_file(const std::string& filename) {
    CSRMatrix A = {0, nullptr, nullptr, nullptr, 0};
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open " + filename);
    }

    std::string line;
    while (std::getline(file, line) && line[0] == '%') {}

    std::stringstream ss(line);
    int n, m, nz;
    ss >> n >> m >> nz;
    if (n != m) {
        throw std::runtime_error("Matrix must be square");
    }
    A.n = n;

    Entry* entries = new Entry[nz]; // Changed: No doubling initially
    int* nnz_per_row = new int[n]();
    int entry_count = 0;

    try {
        for (int k = 0; k < nz; ++k) {
            if (!std::getline(file, line)) {
                throw std::runtime_error("Unexpected end of file");
            }
            ss.clear();
            ss.str(line);
            int i, j;
            flx::floatx<5, 2> val;
            ss >> i >> j >> val;
            if (i < 1 || j < 1 || i > n || j > n) {
                throw std::runtime_error("Invalid indices in Matrix Market file");
            }
            i--; j--;
            entries[entry_count++] = {i, j, val};
            nnz_per_row[i]++;
            // Changed: Do not flx::floatx<5, 2> off-diagonal entries
            // Assume psmigr_2 provides lower triangle; we'll check symmetry later
        }
    } catch (...) {
        delete[] entries;
        delete[] nnz_per_row;
        throw;
    }

    A.nnz = entry_count;
    A.values = new double[A.nnz]();
    A.col_indices = new int[A.nnz]();
    A.row_ptr = new int[n + 1]();
    if (!A.values || !A.col_indices || !A.row_ptr) {
        delete[] entries;
        delete[] nnz_per_row;
        delete[] A.values;
        delete[] A.col_indices;
        delete[] A.row_ptr;
        throw std::runtime_error("Memory allocation failed");
    }

    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row[i];
    }

    std::sort(entries, entries + entry_count,
        [](const Entry& a, const Entry& b) {
            return a.row == b.row ? a.col < b.col : a.row < b.row;
        });

    for (int k = 0; k < A.nnz; ++k) {
        A.col_indices[k] = entries[k].col;
        A.values[k] = entries[k].val;
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << A.nnz << " non-zeros" << std::endl;

    delete[] entries;
    delete[] nnz_per_row;
    return A;
}

void free_csr_matrix(CSRMatrix& A) {
    if (A.values) { delete[] A.values; A.values = nullptr; }
    if (A.col_indices) { delete[] A.col_indices; A.col_indices = nullptr; }
    if (A.row_ptr) { delete[] A.row_ptr; A.row_ptr = nullptr; }
}

void free_result(Result& result) {
    if (result.x) { delete[] result.x; result.x = nullptr; }
    if (result.residual_history) { delete[] result.residual_history; result.residual_history = nullptr; }
}

void matvec(const CSRMatrix& A, const double* x, double* y) {
    #pragma omp parallel for schedule(dynamic) if (A.n > 1000)
    for (int i = 0; i < A.n; ++i) {
        y[i] = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
}

double* generate_rhs(const CSRMatrix& A) {
    double* x_true = new double[A.n]();
    double* b = new double[A.n]();
    if (!x_true || !b) {
        delete[] x_true;
        delete[] b;
        throw std::runtime_error("Memory allocation failed");
    }
    for (int i = 0; i < A.n; ++i) {
        x_true[i] = 1.0;
    }
    matvec(A, x_true, b);
    std::cout << "Generated b = A * x_true, where x_true = [1, 1, ..., 1]" << std::endl;
    delete[] x_true;
    return b;
}

float dot(const double* a, const double* b, int n) {
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum) if (n > 1000)
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void axpy(float alpha, const double* x, const double* y, int n, double* result) {
    #pragma omp parallel for if (n > 1000)
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
}

float norm(const double* v, int n) {
    float d = dot(v, v, n);
    if (isnan(d) || isinf(d) || d < 0.0) {
        throw std::runtime_error("Invalid norm");
    }
    return sqrt(d);
}

float compute_forward_error(const double* x, int n) {
    double* x_true = new double[n]();
    if (!x_true) throw std::runtime_error("Memory allocation failed for x_true");
    
    for (int i = 0; i < n; ++i) {
        x_true[i] = 1.0;
    }
    
    double* error = new double[n]();
    if (!error) {
        delete[] x_true;
        throw std::runtime_error("Memory allocation failed for error");
    }
    
    for (int i = 0; i < n; ++i) {
        error[i] = x[i] - x_true[i];
    }
    
    float forward_error = norm(error, n);
    
    delete[] x_true;
    delete[] error;
    return forward_error;
}

// New: Diagonal (Jacobi) preconditioner
void apply_diag_precond(const CSRMatrix& A, const double* r, double* z, int n) {
    std::vector<float> diag(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] == i) {
                diag[i] = A.values[j];
                break;
            }
        }
        if (abs(diag[i]) < 1e-10) diag[i] = 1.0; // Fallback for zero diagonal
    }
    #pragma omp parallel for if (n > 1000)
    for (int i = 0; i < n; ++i) {
        z[i] = r[i] / diag[i];
    }
}


void arnoldi_step(const CSRMatrix& A, const double* r, double* V, double* H, int j, int n,
                  double* w, double* z, float initial_norm, int restart) {
    apply_diag_precond(A, &V[j * n], z, n); // Changed: Use diagonal preconditioner
    matvec(A, z, w);
    for (int i = 0; i <= j; ++i) {
        float h_ij = dot(w, &V[i * n], n);
        H[i * restart + j] = h_ij;
        axpy(-h_ij, &V[i * n], w, n, w);
    }
    double h_jp1_j = norm(w, n);
    double check_point = 1e-12;
    H[(j + 1) * restart + j] = h_jp1_j;
    if (h_jp1_j < check_point * initial_norm) {
        // Changed: Warn but continue with normalization
        std::cerr << "Warning: Small h_jp1_j at iteration " << j << ", continuing..." << std::endl;
        h_jp1_j = max(h_jp1_j, check_point); // Prevent division by zero
    }
    #pragma omp parallel for if (n > 1000)
    for (int i = 0; i < n; ++i) {
        V[(j + 1) * n + i] = w[i] / h_jp1_j;
    }
}

Result gmres(const CSRMatrix& A, const double* b, int max_iter, double tol, int restart) {
    int n = A.n;
    Result result = {new double[n](), 0.0, 0, nullptr, 0};
    if (!result.x) throw std::runtime_error("Memory allocation failed");

    if (restart > n || restart <= 0) {
        free_result(result);
        throw std::runtime_error("Invalid restart parameter");
    }
    if (max_iter <= 0) {
        free_result(result);
        throw std::runtime_error("Invalid max_iter parameter");
    }

    double* r = new double[n]();
    double* residual_history = new double[max_iter + 1]();
    double* V = new double[n * (restart + 1)]();
    double* H = new double[(restart + 1) * restart]();
    double* w = new double[n]();
    double* z = new double[n]();
    double* g = new double[restart + 1]();
    double* cs = new double[restart]();
    double* sn = new double[restart]();
    int residual_history_size = 0;

    if (!r || !residual_history || !V || !H || !w || !z || !g || !cs || !sn) {
        delete[] r; delete[] residual_history; delete[] V; delete[] H; delete[] w;
        delete[] z; delete[] g; delete[] cs; delete[] sn;
        free_result(result);
        throw std::runtime_error("Memory allocation failed");
    }

    struct Cleanup {
        double* r; double* residual_history; double* V; double* H; double* w;
        double* z; double* g; double* cs; double* sn;
        ~Cleanup() {
            delete[] r; delete[] residual_history; delete[] V; delete[] H; delete[] w;
            delete[] z; delete[] g; delete[] cs; delete[] sn;
        }
    } cleanup = {r, residual_history, V, H, w, z, g, cs, sn};

    std::copy(b, b + n, r);
    double initial_norm = norm(r, n);
    std::cout << "Initial norm of residual: " << initial_norm << std::endl;
    double check_point = 1e-16;
    double tol_abs = tol * max(initial_norm, check_point);

    int total_iterations = 0;
    while (total_iterations < max_iter) {
        matvec(A, result.x, r);
        axpy(-1.0, r, b, n, r);
        float r_norm = norm(r, n);
        if (residual_history_size < max_iter + 1) {
            residual_history[residual_history_size++] = r_norm;
        }
        if (r_norm < tol_abs || r_norm < 0 || isnan(r_norm) || isinf(r_norm)) {
            break;
        }

        for (int i = 0; i < n; ++i) V[i] = r[i] / r_norm;
        g[0] = r_norm;
        std::fill(g + 1, g + restart + 1, 0.0);
        std::fill(H, H + (restart + 1) * restart, 0.0);

        int j = 0;
        bool breakdown = false;
        try {
            for (j = 0; j < restart && total_iterations < max_iter; ++j) {
                arnoldi_step(A, r, V, H, j, n, w, z, initial_norm, restart); // Changed: Pass r

                for (int i = 0; i < j; ++i) {
                    float temp = cs[i] * H[i * restart + j] + sn[i] * H[(i + 1) * restart + j];
                    H[(i + 1) * restart + j] = -sn[i] * H[i * restart + j] + cs[i] * H[(i + 1) * restart + j];
                    H[i * restart + j] = temp;
                }
                float a = H[j * restart + j];
                float b1 = H[(j + 1) * restart + j];
                float rho = sqrt(a * a + b1 * b1);
                if (rho < 1e-12 * initial_norm) {
                    std::cerr << "Warning: Givens rotation breakdown at iteration " << total_iterations << std::endl;
                    breakdown = true;
                    break;
                }
                cs[j] = a / rho;
                sn[j] = b1 / rho;
                H[j * restart + j] = rho;
                H[(j + 1) * restart + j] = 0.0;

                float temp = cs[j] * g[j] + sn[j] * g[j + 1];
                g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
                g[j] = temp;

                r_norm = abs(g[j + 1]);
                if (residual_history_size < max_iter + 1) {
                    residual_history[residual_history_size++] = r_norm;
                }
                total_iterations++;

                // Changed: Log every 10 iterations
                if (total_iterations % 10 == 0) {
                    std::cout << "Iteration " << total_iterations << ": Residual = " << r_norm << std::endl;
                }
                if (r_norm < tol_abs) {
                    j++;
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
            breakdown = true;
        }

        float* y = new float[j]();
        if (!y) throw std::runtime_error("Memory allocation failed");
        for (int i = j - 1; i >= 0 && !breakdown; --i) {
            y[i] = g[i];
            for (int k = i + 1; k < j; ++k) {
                y[i] -= H[i * restart + k] * y[k];
            }
            if (abs(H[i * restart + i]) < 1e-12 * initial_norm) {
                std::cerr << "Warning: Least-squares breakdown at iteration " << total_iterations << std::endl;
                breakdown = true;
                break;
            }
            y[i] /= H[i * restart + i];
        }

        if (!breakdown) {
            for (int k = 0; k < j; ++k) {
                apply_diag_precond(A, &V[k * n], z, n); // Changed: Use diagonal preconditioner
                axpy(y[k], z, result.x, n, result.x);
            }
        }
        delete[] y;

        if (r_norm < tol_abs || breakdown) {
            break;
        }
    }

    matvec(A, result.x, r);
    axpy(-1.0, r, b, n, r);
    float r_norm = norm(r, n);
    if (residual_history_size < max_iter + 1) {
        residual_history[residual_history_size++] = r_norm;
    }

    result.residual = r_norm;
    result.iterations = total_iterations;
    result.residual_history = residual_history;
    result.residual_history_size = residual_history_size;

    cleanup.residual_history = nullptr;
    return result;
}

int main(int argc, char* argv[]) {
    CSRMatrix A = {0, nullptr, nullptr, nullptr, 0};
    double* b = nullptr;
    Result result = {nullptr, 0.0, 0, nullptr, 0};

    try {
        std::string filename = (argc > 1) ? argv[1] : "psmigr_2.mtx";
        A = read_mtx_file(filename);
        b = generate_rhs(A);

        std::cout << "A.n=" << A.n << ", A.nnz=" << A.nnz << std::endl;

        // Changed: Looser tolerance, smaller restart
        int max_iter = (argc > 2) ? std::stoi(argv[2]) : A.n;
        double tol = (argc > 3) ? std::stod(argv[3]) : 1e-10;
        int restart = (argc > 4) ? std::stoi(argv[4]) : 100;

        auto start = std::chrono::high_resolution_clock::now();
        result = gmres(A, b, max_iter, tol, restart);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        flx::floatx<5, 2> *check_x = new flx::floatx<5, 2>[A.n];
        for (int i = 0; i < A.n; ++i) {
            check_x[i] = result.x[i];
        }
        
        PROMISE_CHECK_ARRAY(check_x, A.n);
        double forward_error = compute_forward_error(result.x, A.n);

        std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
        std::cout << "Training time: " << duration.count() << " ms" << std::endl;
        std::cout << "Final residual: " << result.residual << std::endl;
        std::cout << "Forward error (||x - x_true||): " << forward_error << std::endl;
        std::cout << "Iterations to converge: " << result.iterations << std::endl;

        double* r = new double[A.n]();
        if (!r) throw std::runtime_error("Memory allocation failed");
        matvec(A, result.x, r);
        axpy(-1.0, r, b, A.n, r);
        double verify_residual = norm(r, A.n);
        std::cout << "Verification residual: " << verify_residual << std::endl;
        delete[] r;

        std::cout << "First 5 solution entries:\n";
        for (int i = 0; i < min(5, A.n); ++i) {
            std::cout << "x[" << i << "] = " << result.x[i] << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        free_csr_matrix(A);
        delete[] b;
        free_result(result);
        return 1;
    }

    free_csr_matrix(A);
    delete[] b;
    free_result(result);
    return 0;
}