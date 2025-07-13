#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <cstring>

struct CSRMatrix {
    int n;
    double* values;
    int* col_indices;
    int* row_ptr;
    int nnz;
};

struct Entry { int row, col; double val; };

struct Result {
    double* x;
    float residual;
    int iterations;
    double* residual_history;
    int residual_history_size;
};

void free_csr_matrix(CSRMatrix& A) {
    if (A.values) { delete[] A.values; A.values = nullptr; }
    if (A.col_indices) { delete[] A.col_indices; A.col_indices = nullptr; }
    if (A.row_ptr) { delete[] A.row_ptr; A.row_ptr = nullptr; }
}

void free_result(Result& res) {
    if (res.x) { delete[] res.x; res.x = nullptr; }
    if (res.residual_history) { delete[] res.residual_history; res.residual_history = nullptr; }
}

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

    for (int k = 0; k < nz; ++k) {
        if (!std::getline(file, line)) {
            delete[] entries;
            delete[] nnz_per_row;
            throw std::runtime_error("Unexpected end of file");
        }
        ss.clear();
        ss.str(line);
        int i, j;
        double val;
        ss >> i >> j >> val;
        if (i < 1 || j < 1 || i > n || j > n) {
            delete[] entries;
            delete[] nnz_per_row;
            throw std::runtime_error("Invalid indices in Matrix Market file");
        }
        i--; j--;
        entries[entry_count++] = {i, j, val};
        if (i != j) entries[entry_count++] = {j, i, val};
        nnz_per_row[i]++;
        if (i != j) nnz_per_row[j]++;
    }

    A.nnz = entry_count;
    A.values = new double[A.nnz];
    A.col_indices = new int[A.nnz];
    A.row_ptr = new int[n + 1];
    if (!A.values || !A.col_indices || !A.row_ptr) {
        delete[] entries;
        delete[] nnz_per_row;
        free_csr_matrix(A);
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

    delete[] entries;
    delete[] nnz_per_row;
    std::cout << "Loaded matrix: " << n << " x " << n << " with " << A.nnz << " non-zeros" << std::endl;
    return A;
}

void matvec(const CSRMatrix& A, const double* x, double* y) {
    for (int i = 0; i < A.n; ++i) {
        y[i] = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
}

double* generate_rhs(const CSRMatrix& A) {
    double* x_true = new double[A.n];
    double* b = new double[A.n];
    if (!x_true || !b) {
        delete[] x_true;
        delete[] b;
        throw std::runtime_error("Memory allocation failed");
    }
    for (int i = 0; i < A.n; ++i) {
        x_true[i] = 1.0;
        b[i] = 0.0;
    }
    matvec(A, x_true, b);
    delete[] x_true;
    std::cout << "Generated b = A * x_true, where x_true = [1, 1, ..., 1]" << std::endl;
    return b;
}

double dot(const double* a, const double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void axpy(double alpha, const double* x, const double* y, int n, double* result) {
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
}

double norm(const double* v, int n) {
    double d = dot(v, v, n);

    return sqrt(d);
}

double* compute_diagonal_preconditioner(const CSRMatrix& A) {
    double* M = new double[A.n]();
    if (!M) throw std::runtime_error("Memory allocation failed");
    double min_diag = 9999.9;
    double max_diag = 0.0;
    bool has_zero_diagonal = false;

    for (int i = 0; i < A.n; ++i) {
        bool found_diag = false;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] == i) {
                M[i] = A.values[j];
                found_diag = true;
                min_diag = min(min_diag, abs(M[i]));
                max_diag = max(max_diag, abs(M[i]));
                break;
            }
        }
        if (!found_diag || abs(M[i]) < 1e-10) {
            has_zero_diagonal = true;
            M[i] = 1.0; // Default for zero or missing diagonal
        } else {
            M[i] = 1.0 / M[i];
        }
    }
    if (has_zero_diagonal) {
        std::cerr << "Warning: Matrix has zero or near-zero diagonal elements" << std::endl;
    }
    std::cout << "Diagonal stats: min |A_ii| = " << min_diag << ", max |A_ii| = " << max_diag << std::endl;
    return M;
}

void apply_preconditioner(const double* M, const double* r, int n, double* z) {
    for (int i = 0; i < n; ++i) {
        z[i] = M[i] * r[i];
    }
}


void arnoldi_step(const CSRMatrix& A, const double* M, double* V, double* H, int j,
                  int n, double* z, double* w, float initial_norm, int restart) {
    apply_preconditioner(M, &V[j * n], n, z);
    matvec(A, z, w);
    for (int i = 0; i <= j; ++i) {
        double h_ij = dot(w, &V[i * n], n);
        H[i * restart + j] = h_ij;
        axpy(-h_ij, &V[i * n], w, n, w);
    }
    double h_jp1_j = norm(w, n);
    H[(j + 1) * restart + j] = h_jp1_j;
    if (h_jp1_j < 1e-12 * initial_norm) {
        throw std::runtime_error("Arnoldi breakdown at iteration " + std::to_string(j));
    }
    for (int i = 0; i < n; ++i) {
        V[(j + 1) * n + i] = w[i] / h_jp1_j;
    }
}

Result gmres(const CSRMatrix& A, const double* b, int max_iter, float tol, int restart) {
    int n = A.n;
    Result result = {new double[n](), 0.0, 0, nullptr, 0};
    if (!result.x) throw std::runtime_error("Memory allocation failed");

    // Validate inputs
    if (restart > n || restart <= 0) {
        free_result(result);
        throw std::runtime_error("Invalid restart parameter");
    }
    if (max_iter <= 0) {
        free_result(result);
        throw std::runtime_error("Invalid max_iter parameter");
    }

    // Allocate temporary arrays
    double* r = new double[n]();
    double* M = compute_diagonal_preconditioner(A);
    double* V = new double[n * (restart + 1)]();
    double* H = new double[(restart + 1) * restart]();
    double* z = new double[n]();
    double* w = new double[n]();
    double* g = new double[restart + 1]();
    double* cs = new double[restart]();
    double* sn = new double[restart]();
    double* residual_history = new double[max_iter + 1]();
    int residual_history_size = 0;

    if (!r || !M || !V || !H || !z || !w || !g || !cs || !sn || !residual_history) {
        delete[] r; delete[] M; delete[] V; delete[] H; delete[] z; delete[] w;
        delete[] g; delete[] cs; delete[] sn; delete[] residual_history;
        free_result(result);
        throw std::runtime_error("Memory allocation failed");
    }

    // Scope-based cleanup for temporary arrays
    struct Cleanup {
        double* r; double* M; double* V; double* H; double* z; double* w;
        double* g; double* cs; double* sn; double* residual_history;
        ~Cleanup() {
            delete[] r; delete[] M; delete[] V; delete[] H; delete[] z; delete[] w;
            delete[] g; delete[] cs; delete[] sn; delete[] residual_history;
        }
    } cleanup = {r, M, V, H, z, w, g, cs, sn, residual_history};

    // Initial residual: r = b
    std::copy(b, b + n, r);
    double initial_norm = norm(r, n);
    std::cout << "Initial norm of residual: " << initial_norm << std::endl;
    double temp = 1e-10;
    float tol_abs = tol * max(initial_norm, temp);

    int total_iterations = 0;
    while (total_iterations < max_iter) {
        // Compute true residual: r = b - A*x
        matvec(A, result.x, r);
        axpy(-1.0, r, b, n, r);
        double r_norm = norm(r, n);
        if (residual_history_size < max_iter + 1) {
            residual_history[residual_history_size++] = r_norm;
        }
        if (r_norm < tol_abs || r_norm < 0 || isnan(r_norm) || isinf(r_norm)) {
            break;
        }

        // Initialize V[:,0]
        for (int i = 0; i < n; ++i) V[i] = r[i] / r_norm;
        g[0] = r_norm;
        std::fill(g + 1, g + restart + 1, 0.0);
        std::fill(H, H + (restart + 1) * restart, 0.0);

        int j = 0;
        bool breakdown = false;
        try {
            for (j = 0; j < restart && total_iterations < max_iter; ++j) {
                arnoldi_step(A, M, V, H, j, n, z, w, initial_norm, restart);

                // Apply Givens rotations
                for (int i = 0; i < j; ++i) {
                    double temp = cs[i] * H[i * restart + j] + sn[i] * H[(i + 1) * restart + j];
                    H[(i + 1) * restart + j] = -sn[i] * H[i * restart + j] + cs[i] * H[(i + 1) * restart + j];
                    H[i * restart + j] = temp;
                }
                double a = H[j * restart + j];
                double b1 = H[(j + 1) * restart + j];
                double rho = sqrt(a * a + b1 * b1);
                if (rho < 1e-12 * initial_norm) {
                    throw std::runtime_error("Givens rotation breakdown");
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

        // Solve least-squares problem
        double* y = new double[j]();
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

        // Update solution
        if (!breakdown) {
            for (int k = 0; k < j; ++k) {
                apply_preconditioner(M, &V[k * n], n, z);
                axpy(y[k], z, result.x, n, result.x);
            }
        }
        delete[] y;

        if (r_norm < tol_abs || breakdown) {
            break;
        }
    }

    // Final residual
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

    // Prevent cleanup from deleting residual_history (transferred to result)
    cleanup.residual_history = nullptr;
    return result;
}

int main(int argc, char* argv[]) {
    CSRMatrix A = {0, nullptr, nullptr, nullptr, 0};
    double* b = nullptr;
    Result result = {nullptr, 0.0, 0, nullptr, 0};

    try {
        std::string filename = "1138_bus.mtx";
        int max_iter = 1000;
        float tol = 1e-8;
        int restart = 1000;

        A = read_mtx_file(filename);
        b = generate_rhs(A);

        auto start = std::chrono::high_resolution_clock::now();
        result = gmres(A, b, max_iter, tol, restart);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        double* r = new double[A.n];
        if (!r) throw std::runtime_error("Memory allocation failed");
        matvec(A, result.x, r);
        axpy(-1.0, r, b, A.n, r);
        float verify_residual = norm(r, A.n);
        std::cout << "Verification residual: " << verify_residual << std::endl;
        delete[] r;

        double check_solution[A.n];
        for (int i = 0; i < A.n; ++i) {
            check_solution[i] = result.x[i];
        }

        PROMISE_CHECK_ARRAY(check_solution, A.n);

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