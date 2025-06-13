#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <omp.h>

struct CSRMatrix {
    int n;
    double* values;
    int* col_indices;
    int* row_ptr;
    int nnz;
};

struct CholeskyPreconditioner {
    int n;
    double* L_values;
    int* L_col_indices;
    int* L_row_ptr;
    int nnz;
    bool is_valid;
};

bool compare_by_column(const std::pair<int, double>& a, const std::pair<int, double>& b) {
    return a.first < b.first;
}

CSRMatrix read_mtx_file(const std::string& filename) {
    CSRMatrix A = {0, nullptr, nullptr, nullptr, 0};
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return A;
    }

    std::string line;
    while (getline(file, line) && line[0] == '%') {}

    std::stringstream ss(line);
    int n, m, nz;
    ss >> n >> m >> nz;
    if (n != m) {
        std::cerr << "Error: Matrix must be square" << std::endl;
        return A;
    }
    A.n = n;

    std::vector<std::vector<std::pair<int, double>>> rows(n);
    int entry_count = 0;
    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        double val;
        ss >> i >> j >> val;
        if (i < 1 || j < 1 || i > n || j > n) {
            std::cerr << "Error: Invalid indices at line " << k + 1 << std::endl;
            return A;
        }
        i--; j--;
        rows[i].emplace_back(j, val);
        if (i != j) {
            rows[j].emplace_back(i, val);
            entry_count++;
        }
        entry_count++;
    }

    std::vector<int> nnz_per_row(n, 0);
    for (int i = 0; i < n; ++i) {
        std::sort(rows[i].begin(), rows[i].end(), compare_by_column);
        nnz_per_row[i] = rows[i].size();
    }

    A.nnz = entry_count;
    A.values = new double[entry_count];
    A.col_indices = new int[entry_count];
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row[i];
    }

    int idx = 0;
    for (int i = 0; i < n; ++i) {
        for (const auto& entry : rows[i]) {
            A.col_indices[idx] = entry.first;
            A.values[idx] = entry.second;
            idx++;
        }
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << entry_count << " non-zeros" << std::endl;

    double min_diag = std::numeric_limits<double>::max();
    double max_diag = 0.0;
    bool all_diags_positive = true;
    for (int i = 0; i < n; ++i) {
        bool found_diag = false;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] == i) {
                double d = A.values[j];
                min_diag = std::min(min_diag, std::abs(d));
                max_diag = std::max(max_diag, std::abs(d));
                if (d <= 0) all_diags_positive = false;
                found_diag = true;
                break;
            }
        }
        if (!found_diag) {
            std::cerr << "Warning: No diagonal entry for row " << i << std::endl;
            all_diags_positive = false;
        }
    }
    std::cout << "Matrix diagonal stats: min |A_ii| = " << min_diag << ", max |A_ii| = " << max_diag << std::endl;
    if (!all_diags_positive) {
        std::cerr << "Warning: Matrix may not be positive definite due to non-positive diagonal entries" << std::endl;
    }

    return A;
}

void free_csr_matrix(CSRMatrix& A) {
    delete[] A.values;
    delete[] A.col_indices;
    delete[] A.row_ptr;
    A.values = nullptr;
    A.col_indices = nullptr;
    A.row_ptr = nullptr;
}

void free_cholesky_preconditioner(CholeskyPreconditioner& P) {
    delete[] P.L_values;
    delete[] P.L_col_indices;
    delete[] P.L_row_ptr;
    P.L_values = nullptr;
    P.L_col_indices = nullptr;
    P.L_row_ptr = nullptr;
}

void matvec(const CSRMatrix& A, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < A.n; ++i) {
        y[i] = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
}

double dot(const double* a, const double* b, int n) {
    double sum = 0.0, c = 0.0;
    #pragma omp parallel for reduction(+:sum, c)
    for (int i = 0; i < n; ++i) {
        double y = a[i] * b[i] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

void axpy(double alpha, const double* x, const double* y, int n, double* result) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
}

double norm(const double* v, int n) {
    double d = dot(v, v, n);
    if (std::isnan(d) || std::isinf(d)) return -1.0;
    return std::sqrt(d);
}

void compute_diagonal_preconditioner(const CSRMatrix& A, double* M) {
    bool has_zero_diagonal = false;
    double min_diag = std::numeric_limits<double>::max();
    double max_diag = 0.0;
    for (int i = 0; i < A.n; ++i) {
        M[i] = 0.0;
        bool found_diag = false;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] == i) {
                M[i] = A.values[j];
                found_diag = true;
                min_diag = std::min(min_diag, std::abs(M[i]));
                max_diag = std::max(max_diag, std::abs(M[i]));
                break;
            }
        }
        if (!found_diag || std::abs(M[i]) < 1e-10) {
            has_zero_diagonal = true;
            M[i] = 1.0;
        } else {
            M[i] = 1.0 / M[i];
        }
    }
    if (has_zero_diagonal) {
        std::cerr << "Warning: Matrix has zero or near-zero diagonal elements" << std::endl;
    }
    std::cout << "Diagonal preconditioner stats: min |M_ii| = " << min_diag << ", max |M_ii| = " << max_diag << std::endl;
}

void apply_diagonal_preconditioner(const double* M, const double* r, int n, double* z) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        z[i] = M[i] * r[i];
    }
}

CholeskyPreconditioner compute_ic_preconditioner(const CSRMatrix& A) {
    CholeskyPreconditioner P = {A.n, nullptr, nullptr, nullptr, 0, false};
    std::vector<std::vector<std::pair<int, double>>> lower(A.n);
    
    // Extract lower triangular part
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int k = A.col_indices[j];
            if (k <= i) {
                lower[i].emplace_back(k, A.values[j]);
            }
        }
        std::sort(lower[i].begin(), lower[i].end(), compare_by_column);
    }

    // Count non-zeros in L
    int nnz = 0;
    for (int i = 0; i < A.n; ++i) nnz += lower[i].size();
    P.nnz = nnz;
    P.L_values = new double[nnz];
    P.L_col_indices = new int[nnz];
    P.L_row_ptr = new int[A.n + 1];
    P.L_row_ptr[0] = 0;
    for (int i = 0; i < A.n; ++i) {
        P.L_row_ptr[i + 1] = P.L_row_ptr[i] + lower[i].size();
    }

    int idx = 0;
    for (int i = 0; i < A.n; ++i) {
        for (const auto& entry : lower[i]) {
            P.L_col_indices[idx] = entry.first;
            P.L_values[idx] = 0.0;
            idx++;
        }
    }

    std::vector<double> diag(A.n, 0.0);
    const double epsilon = 1e-10; // Small shift for numerical stability
    for (int i = 0; i < A.n; ++i) {
        double sum = 0.0;
        int diag_idx = -1;
        for (int j = P.L_row_ptr[i]; j < P.L_row_ptr[i + 1]; ++j) {
            int k = P.L_col_indices[j];
            if (k < i) {
                double l_ik = P.L_values[j];
                sum += l_ik * l_ik;
            } else if (k == i) {
                diag_idx = j;
                break;
            }
        }
        double a_ii = 0.0;
        for (const auto& entry : lower[i]) {
            if (entry.first == i) {
                a_ii = entry.second;
                break;
            }
        }
        double d = a_ii - sum + epsilon;
        if (d <= 0) {
            std::cerr << "Warning: Non-positive diagonal at row " << i << " (d = " << d << "), using diagonal preconditioner" << std::endl;
            free_cholesky_preconditioner(P);
            return {0, nullptr, nullptr, nullptr, 0, false};
        }
        P.L_values[diag_idx] = std::sqrt(d);
        diag[i] = P.L_values[diag_idx];

        for (int j = diag_idx + 1; j < P.L_row_ptr[i + 1]; ++j) {
            int k = P.L_col_indices[j];
            double a_ik = 0.0;
            for (const auto& entry : lower[i]) {
                if (entry.first == k) {
                    a_ik = entry.second;
                    break;
                }
            }
            double sum = 0.0;
            for (int m = P.L_row_ptr[i]; m < diag_idx; ++m) {
                int col_m = P.L_col_indices[m];
                for (int n = P.L_row_ptr[k]; n < P.L_row_ptr[k + 1]; ++n) {
                    if (P.L_col_indices[n] == col_m) {
                        sum += P.L_values[m] * P.L_values[n];
                    }
                }
            }
            P.L_values[j] = (a_ik - sum) / diag[i];
        }
    }

    double min_diag = *std::min_element(diag.begin(), diag.end());
    double max_diag = *std::max_element(diag.begin(), diag.end());
    std::cout << "IC(0) diagonal stats: min |L_ii| = " << min_diag << ", max |L_ii| = " << max_diag << std::endl;
    P.is_valid = true;

    return P;
}

void apply_ic_preconditioner(const CholeskyPreconditioner& P, const double* r, double* z) {
    std::vector<double> y(P.n, 0.0);
    // Forward: L y = r
    for (int i = 0; i < P.n; ++i) {
        double sum = 0.0;
        for (int j = P.L_row_ptr[i]; j < P.L_row_ptr[i + 1]; ++j) {
            int k = P.L_col_indices[j];
            if (k < i) {
                sum += P.L_values[j] * y[k];
            } else if (k == i) {
                if (P.L_values[j] != 0.0) {
                    y[i] = (r[i] - sum) / P.L_values[j];
                } else {
                    y[i] = 0.0;
                }
                break;
            }
        }
    }
    // Backward: L^T z = y
    for (int i = P.n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int j = P.L_row_ptr[i]; j < P.L_row_ptr[i + 1]; ++j) {
            int k = P.L_col_indices[j];
            if (k > i) {
                sum += P.L_values[j] * z[k];
            } else if (k == i) {
                if (P.L_values[j] != 0.0) {
                    z[i] = (y[i] - sum) / P.L_values[j];
                } else {
                    z[i] = 0.0;
                }
                break;
            }
        }
    }
}

struct Result {
    double* x;
    double residual;
    int iterations;
    std::vector<double> residual_history;
};

Result pcg(const CSRMatrix& A, const double* b, const CholeskyPreconditioner& P, const double* M, bool use_ic, int max_iter = 1000, double tol = 1e-10) {
    int n = A.n;
    std::vector<double> x(n, 0.0);
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) r[i] = b[i];
    std::vector<double> z(n);
    if (use_ic) {
        apply_ic_preconditioner(P, r.data(), z.data());
    } else {
        apply_diagonal_preconditioner(M, r.data(), n, z.data());
    }
    std::vector<double> p(n);
    for (int i = 0; i < n; ++i) p[i] = z[i];
    std::vector<double> residual_history;

    double r_z = dot(r.data(), z.data(), n);
    double initial_norm = norm(b, n);
    if (initial_norm < 0) {
        std::cerr << "Error: Initial b has invalid norm" << std::endl;
        double* x_out = new double[n];
        for (int i = 0; i < n; ++i) x_out[i] = x[i];
        return {x_out, -1.0, 0, {}};
    }
    std::cout << "Initial norm of b: " << initial_norm << std::endl;
    double tol_abs = tol * initial_norm;
    double prev_r_norm = std::numeric_limits<double>::max();

    int k;
    for (k = 0; k < max_iter; ++k) {
        std::vector<double> Ap(n);
        matvec(A, p.data(), Ap.data());
        double p_Ap = dot(p.data(), Ap.data(), n);
        if (std::abs(p_Ap) < 1e-14) {
            std::cerr << "Breakdown: p^T Ap = " << p_Ap << " at iteration " << k << std::endl;
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            double* r_temp = new double[n];
            matvec(A, x.data(), r_temp);
            axpy(-1.0, r_temp, b, n, r_temp);
            double* z_temp = new double[n];
            if (use_ic) {
                apply_ic_preconditioner(P, r_temp, z_temp);
            } else {
                apply_diagonal_preconditioner(M, r_temp, n, z_temp);
            }
            double r_norm = norm(z_temp, n);
            delete[] r_temp; delete[] z_temp;
            return {x_out, r_norm, k, residual_history};
        }
        double alpha = r_z / p_Ap;
        axpy(alpha, p.data(), x.data(), n, x.data());
        axpy(-alpha, Ap.data(), r.data(), n, r.data());
        if (use_ic) {
            apply_ic_preconditioner(P, r.data(), z.data());
        } else {
            apply_diagonal_preconditioner(M, r.data(), n, z.data());
        }
        double r_norm = norm(z.data(), n);
        residual_history.push_back(r_norm);
        if (r_norm < tol_abs) {
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            return {x_out, r_norm, k + 1, residual_history};
        }
        if (k > 0 && r_norm > prev_r_norm * 0.999) {
            std::cerr << "Stagnation detected: Residual = " << r_norm << " at iteration " << k << std::endl;
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            return {x_out, r_norm, k, residual_history};
        }
        double r_z_new = dot(r.data(), z.data(), n);
        if (std::abs(r_z_new) < 1e-14) {
            std::cerr << "Breakdown: r^T z = " << r_z_new << " at iteration " << k << std::endl;
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            double* r_temp = new double[n];
            matvec(A, x.data(), r_temp);
            axpy(-1.0, r_temp, b, n, r_temp);
            double* z_temp = new double[n];
            if (use_ic) {
                apply_ic_preconditioner(P, r_temp, z_temp);
            } else {
                apply_diagonal_preconditioner(M, r_temp, n, z_temp);
            }
            double r_norm = norm(z_temp, n);
            delete[] r_temp; delete[] z_temp;
            return {x_out, r_norm, k, residual_history};
        }
        double beta = r_z_new / r_z;
        axpy(beta, p.data(), z.data(), n, p.data());
        r_z = r_z_new;
        prev_r_norm = r_norm;
        if (k % 10 == 0) {
            std::cout << "Iteration " << k << ": Preconditioned Residual = " << r_norm << std::endl;
        }
    }

    double* r_temp = new double[n];
    matvec(A, x.data(), r_temp);
    axpy(-1.0, r_temp, b, n, r_temp);
    double* z_temp = new double[n];
    if (use_ic) {
        apply_ic_preconditioner(P, r_temp, z_temp);
    } else {
        apply_diagonal_preconditioner(M, r_temp, n, z_temp);
    }
    double r_norm = norm(z_temp, n);
    delete[] r_temp; delete[] z_temp;
    double* x_out = new double[n];
    for (int i = 0; i < n; ++i) x_out[i] = x[i];
    return {x_out, r_norm, k + 1, residual_history};
}

double* generate_rhs(const CSRMatrix& A) {
    std::vector<double> x_true(A.n, 1.0);
    double* b = new double[A.n];
    matvec(A, x_true.data(), b);
    std::cout << "Generated b = A * x_true, where x_true = [1, 1, ..., 1]" << std::endl;
    return b;
}

void write_solution(const double* x, int n, const std::string& filename, const std::vector<double>& residual_history, const double* x_true) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << ". Check permissions or path." << std::endl;
        return;
    }
    file << "x\n";
    for (int i = 0; i < n; ++i) {
        file << x[i] << "\n";
    }
    file << "\nResidual History\n";
    for (size_t i = 0; i < residual_history.size(); ++i) {
        file << i << "," << residual_history[i] << "\n";
    }
    file.close();

    double error_norm = 0.0;
    #pragma omp parallel for reduction(+:error_norm)
    for (int i = 0; i < n; ++i) {
        double diff = x[i] - x_true[i];
        error_norm += diff * diff;
    }
    error_norm = std::sqrt(error_norm);
    std::cout << "Error norm vs x_true: " << error_norm << std::endl;
}

int main() {
    std::string filename = "../data/suitesparse/1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) {
        free_csr_matrix(A);
        return 1;
    }

    double* b = generate_rhs(A);
    std::vector<double> x_true(A.n, 1.0);
    CholeskyPreconditioner P = compute_ic_preconditioner(A);
    double* M = new double[A.n];
    compute_diagonal_preconditioner(A, M);
    bool use_ic = P.is_valid;

    auto start = std::chrono::high_resolution_clock::now();
    Result result = pcg(A, b, P, M, use_ic, 2 * A.n, 1e-10);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final preconditioned residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    double* r_temp = new double[A.n];
    matvec(A, result.x, r_temp);
    axpy(-1.0, r_temp, b, A.n, r_temp);
    double* z_temp = new double[A.n];
    if (use_ic) {
        apply_ic_preconditioner(P, r_temp, z_temp);
    } else {
        apply_diagonal_preconditioner(M, r_temp, A.n, z_temp);
    }
    double verify_residual = norm(z_temp, A.n);
    std::cout << "Verification preconditioned residual: " << verify_residual << std::endl;
    delete[] r_temp; delete[] z_temp;

    write_solution(result.x, A.n, "pcg_ic_solution.csv", result.residual_history, x_true.data());

    delete[] b;
    delete[] result.x;
    delete[] M;
    free_csr_matrix(A);
    if (use_ic) free_cholesky_preconditioner(P);
    return 0;
}