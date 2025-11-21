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

struct CSRMatrix {
    int n;
    double* values;
    int* col_indices;
    int* row_ptr;
    int nnz;
};

struct Entry { int row, col; flx::floatx<8, 7> val; };

struct Result {
    double* x;
    flx::floatx<4, 3> residual;
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

    Entry* entries = new Entry[2 * nz];
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
            flx::floatx<8, 7> val;
            ss >> i >> j >> val;
            if (i < 1 || j < 1 || i > n || j > n) {
                throw std::runtime_error("Invalid indices in Matrix Market file");
            }
            i--; j--;
            entries[entry_count++] = {i, j, val};
            if (i != j) entries[entry_count++] = {j, i, val};
            nnz_per_row[i]++;
            if (i != j) nnz_per_row[j]++;
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
    double sum = 0.0;
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

flx::floatx<4, 3> compute_forward_error(const double* x, int n) {
    flx::floatx<4, 3>* x_true = new flx::floatx<4, 3>[n]();
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
    
    flx::floatx<4, 3> forward_error = norm(error, n);
    
    delete[] x_true;
    delete[] error;
    return forward_error;
}

CSRMatrix compute_sparse_lu_factorization(const CSRMatrix& A) {
    CSRMatrix LU = {A.n, nullptr, nullptr, nullptr, A.nnz};
    LU.values = new double[A.nnz]();
    LU.col_indices = new int[A.nnz]();
    LU.row_ptr = new int[A.n + 1]();
    if (!LU.values || !LU.col_indices || !LU.row_ptr) {
        free_csr_matrix(LU);
        throw std::runtime_error("Memory allocation failed for sparse LU");
    }
    std::copy(A.values, A.values + A.nnz, LU.values);
    std::copy(A.col_indices, A.col_indices + A.nnz, LU.col_indices);
    std::copy(A.row_ptr, A.row_ptr + A.n + 1, LU.row_ptr);

    bool has_zero_diagonal = false;
    for (int i = 0; i < A.n; ++i) {
        // Find diagonal element
        flx::floatx<8, 7> diag = 0.0;
        int diag_idx = -1;
        for (int j = LU.row_ptr[i]; j < LU.row_ptr[i + 1]; ++j) {
            if (LU.col_indices[j] == i) {
                diag = LU.values[j];
                diag_idx = j;
                break;
            }
        }
        if (diag_idx == -1 || abs(diag) < 1e-15) {
            has_zero_diagonal = true;
            diag = 1.0; // Fallback for zero or missing diagonal
            if (diag_idx != -1) LU.values[diag_idx] = diag;
        }

        // Perform elimination for rows k > i
        for (int k = i + 1; k < A.n; ++k) {
            flx::floatx<8, 7> lik = 0.0;
            int lik_idx = -1;
            for (int j = LU.row_ptr[k]; j < LU.row_ptr[k + 1]; ++j) {
                if (LU.col_indices[j] == i) {
                    lik = LU.values[j] / diag;
                    lik_idx = j;
                    break;
                }
            }
            if (lik_idx == -1) continue;

            // Update row k
            for (int j = LU.row_ptr[k]; j < LU.row_ptr[k + 1]; ++j) {
                if (LU.col_indices[j] <= i) continue;
                for (int m = LU.row_ptr[i]; m < LU.row_ptr[i + 1]; ++m) {
                    if (LU.col_indices[m] == LU.col_indices[j]) {
                        LU.values[j] -= lik * LU.values[m];
                        break;
                    }
                }
            }
            LU.values[lik_idx] = lik;
        }
    }
    if (has_zero_diagonal) {
        std::cerr << "Warning: Matrix has zero or near-zero diagonal elements in sparse LU" << std::endl;
    }
    return LU;
}

void forward_solve(const CSRMatrix& LU, const double* r, double* z, int n) {
    float* temp = new float[n]();
    if (!temp) throw std::runtime_error("Memory allocation failed");
    for (int i = 0; i < n; ++i) {
        temp[i] = r[i];
        for (int j = LU.row_ptr[i]; j < LU.row_ptr[i + 1]; ++j) {
            if (LU.col_indices[j] < i) {
                temp[i] -= LU.values[j] * temp[LU.col_indices[j]];
            }
        }
        z[i] = temp[i]; // L is unit lower triangular
    }
    delete[] temp;
}

void backward_solve(const CSRMatrix& LU, const double* z, double* x, int n) {
    float* temp = new float[n]();
    if (!temp) throw std::runtime_error("Memory allocation failed");
    std::copy(z, z + n, temp);
    for (int i = n - 1; i >= 0; --i) {
        flx::floatx<8, 7> diag = 0.0;
        int diag_idx = -1;
        for (int j = LU.row_ptr[i]; j < LU.row_ptr[i + 1]; ++j) {
            if (LU.col_indices[j] == i) {
                diag = LU.values[j];
                diag_idx = j;
                break;
            }
        }
        if (diag_idx == -1 || abs(diag) < 1e-10) {
            diag = 1.0; // Fallback for zero diagonal
        }
        x[i] = temp[i] / diag;
        for (int j = LU.row_ptr[i]; j < LU.row_ptr[i + 1]; ++j) {
            if (LU.col_indices[j] > i) {
                temp[LU.col_indices[j]] -= LU.values[j] * x[i];
            }
        }
    }
    delete[] temp;
}

void apply_sparse_lu_preconditioner(const CSRMatrix& LU, const double* r, double* z, int n) {
    double* temp = new double[n]();
    if (!temp) throw std::runtime_error("Memory allocation failed");
    forward_solve(LU, r, temp, n);
    backward_solve(LU, temp, z, n);
    delete[] temp;
}


void arnoldi_step(const CSRMatrix& A, const CSRMatrix& LU, double* V, double* H, int j, int n,
                  double* w, double* z, flx::floatx<8, 7> initial_norm, int restart) {
    apply_sparse_lu_preconditioner(LU, &V[j * n], z, n);
    matvec(A, z, w);
    for (int i = 0; i <= j; ++i) {
        float h_ij = dot(w, &V[i * n], n);
        H[i * restart + j] = h_ij;
        axpy(-h_ij, &V[i * n], w, n, w);
    }
    float h_jp1_j = norm(w, n);
    H[(j + 1) * restart + j] = h_jp1_j;
    if (h_jp1_j < 1e-12 * initial_norm) {
        throw std::runtime_error("Arnoldi breakdown at iteration " + std::to_string(j));
    }
    #pragma omp parallel for if (n > 1000)
    for (int i = 0; i < n; ++i) {
        V[(j + 1) * n + i] = w[i] / h_jp1_j;
    }
}

Result gmres(const CSRMatrix& A, const double* b, int max_iter, flx::floatx<4, 3> tol, int restart) {
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
    CSRMatrix LU = compute_sparse_lu_factorization(A);
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
        free_csr_matrix(LU);
        free_result(result);
        throw std::runtime_error("Memory allocation failed");
    }

    struct Cleanup {
        double* r; double* residual_history; double* V; double* H; double* w;
        double* z; double* g; double* cs; double* sn; CSRMatrix* LU;
        ~Cleanup() {
            delete[] r; delete[] residual_history; delete[] V; delete[] H; delete[] w;
            delete[] z; delete[] g; delete[] cs; delete[] sn;
            free_csr_matrix(*LU);
        }
    } cleanup = {r, residual_history, V, H, w, z, g, cs, sn, &LU};

    std::copy(b, b + n, r);
    flx::floatx<8, 7> initial_norm = norm(r, n);
    std::cout << "Initial norm of residual: " << initial_norm << std::endl;
    flx::floatx<4, 3> tol_abs = tol;// tol * std::max(initial_norm, 1e-16);

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
                arnoldi_step(A, LU, V, H, j, n, w, z, initial_norm, restart);

                for (int i = 0; i < j; ++i) {
                    float temp = cs[i] * H[i * restart + j] + sn[i] * H[(i + 1) * restart + j];
                    H[(i + 1) * restart + j] = -sn[i] * H[i * restart + j] + cs[i] * H[(i + 1) * restart + j];
                    H[i * restart + j] = temp;
                }
                float a = H[j * restart + j];
                float b1 = H[(j + 1) * restart + j];
                float rho = sqrt(a * a + b1 * b1);
                if (rho < 1e-12 * initial_norm) {
                    throw std::runtime_error("Givens rotation breakdown");
                }
                cs[j] = a / rho;
                sn[j] = b1 / rho;
                H[j * restart + j] = rho;
                H[(j + 1) * restart + j] = 0.0;

                flx::floatx<8, 7> temp = cs[j] * g[j] + sn[j] * g[j + 1];
                g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
                g[j] = temp;

                r_norm = abs(g[j + 1]);
                if (residual_history_size < max_iter + 1) {
                    residual_history[residual_history_size++] = r_norm;
                }
                total_iterations++;

                if (total_iterations % 100 == 0) {
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
                apply_sparse_lu_preconditioner(LU, &V[k * n], z, n);
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
    flx::floatx<4, 3> r_norm = norm(r, n);
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
        std::string filename = (argc > 1) ? argv[1] : "1138_bus.mtx";

        A = read_mtx_file(filename);
        b = generate_rhs(A);

        std::cout << "A.n=" << A.n << ", A.nnz=" << A.nnz << std::endl;


        int max_iter_param = A.n;
        int restart_param = 500;

        int max_iter = (argc > 2) ? std::stoi(argv[2]) : max_iter_param;
        flx::floatx<4, 3> tol = (argc > 3) ? std::stod(argv[3]) : 1e-10;
        int restart = (argc > 4) ? std::stoi(argv[4]) : restart_param;

        auto start = std::chrono::high_resolution_clock::now();
        result = gmres(A, b, max_iter, tol, restart);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        flx::floatx<4, 3> forward_error = compute_forward_error(result.x, A.n);

        std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
        std::cout << "Training time: " << duration.count() << " ms" << std::endl;
        std::cout << "Final residual: " << result.residual << std::endl;
        std::cout << "Forward error (||x - x_true||): " << forward_error << std::endl;
        std::cout << "Iterations to converge: " << result.iterations << std::endl;

        flx::floatx<4, 3> *check_solution = new flx::floatx<4, 3>[A.n]();
        if (!check_solution) throw std::runtime_error("Memory allocation failed");
        for (int i = 0; i < A.n; ++i) {
            check_solution[i] = result.x[i];
        }

        PROMISE_CHECK_ARRAY(check_solution, A.n);

        double* r = new double[A.n]();
        if (!r) throw std::runtime_error("Memory allocation failed");
        matvec(A, result.x, r);
        axpy(-1.0, r, b, A.n, r);
        double verify_residual = norm(r, A.n);
        std::cout << "Verification residual: " << verify_residual << std::endl;
        delete[] r;

        std::cout << "First 5 solution entries:\n";
        for (int i = 0; i < std::min(5, A.n); ++i) {
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