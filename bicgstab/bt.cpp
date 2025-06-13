#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

struct CSRMatrix {
    int n;
    double* values;
    int* col_indices;
    int* row_ptr;
    int nnz; // number of non-zeros
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

    struct Entry { int row, col; double val; };
    std::vector<Entry> entries(2 * nz);
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
            std::cerr << "Error: Invalid indices in Matrix Market file" << std::endl;
            return A;
        }
        i--; j--;
        entries[entry_count++] = {i, j, val};
        if (i != j) entries[entry_count++] = {j, i, val};
    }
    entries.resize(entry_count);

    std::vector<int> nnz_per_row(n, 0);
    for (int k = 0; k < entry_count; ++k) {
        nnz_per_row[entries[k].row]++;
    }

    A.nnz = entry_count;
    A.values = new double[entry_count];
    A.col_indices = new int[entry_count];
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row[i];
    }

    std::sort(entries.begin(), entries.end(),
        [](const Entry& a, const Entry& b) {
            return a.row == b.row ? a.col < b.col : a.row < b.row;
        });

    for (int k = 0; k < entry_count; ++k) {
        A.col_indices[k] = entries[k].col;
        A.values[k] = entries[k].val;
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << entry_count << " non-zeros" << std::endl;

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

double* matvec(const CSRMatrix& A, const double* x) {
    double* y = new double[A.n]();
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return y;
}

double dot(const double* a, const double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double* axpy(double alpha, const double* x, const double* y, int n) {
    double* result = new double[n];
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
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
            M[i] = 1.0; // Fallback for zero or missing diagonal
        } else {
            M[i] = 1.0 / M[i];
        }
    }
    if (has_zero_diagonal) {
        std::cerr << "Warning: Matrix has zero or near-zero diagonal elements" << std::endl;
    }
    std::cout << "Diagonal stats: min |A_ii| = " << min_diag << ", max |A_ii| = " << max_diag << std::endl;
}

void apply_preconditioner(const double* M, const double* r, int n, double* z) {
    for (int i = 0; i < n; ++i) {
        z[i] = M[i] * r[i];
    }
}

struct Result {
    double* x;
    double residual;
    int iterations;
    std::vector<double> residual_history;
};

Result bicgstab(const CSRMatrix& A, const double* b, int max_iter = 1000, double tol = 1e-12) {
    int n = A.n;
    std::vector<double> x(n, 0.0);
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) r[i] = b[i]; // r = b - Ax (x = 0)
    std::vector<double> r_hat(n);
    std::vector<double> p(n);
    std::vector<double> z(n);
    std::vector<double> M(n);
    compute_diagonal_preconditioner(A, M.data());
    apply_preconditioner(M.data(), r.data(), n, r_hat.data()); // r_hat = M^{-1} r_0
    for (int i = 0; i < n; ++i) p[i] = r_hat[i]; // p_0 = M^{-1} r_0
    std::vector<double> v(n, 0.0);
    std::vector<double> residual_history;

    double rho = 1.0, alpha = 1.0, omega = 1.0;
    double initial_norm = norm(b, n);
    if (initial_norm < 0) {
        std::cerr << "Error: Initial b has invalid norm" << std::endl;
        double* x_out = new double[n];
        for (int i = 0; i < n; ++i) x_out[i] = x[i];
        return {x_out, -1.0, 0, {}};
    }
    std::cout << "Initial norm of b: " << initial_norm << std::endl;
    double tol_abs = tol * initial_norm;

    int k;
    for (k = 0; k < max_iter; ++k) {
        // Compute preconditioned residual: r_tilde = M^{-1} r
        std::vector<double> r_tilde(n);
        apply_preconditioner(M.data(), r.data(), n, r_tilde.data());
        double rho_new = dot(r_hat.data(), r_tilde.data(), n);
        if (std::abs(rho_new) < 1e-10) {
            std::cerr << "Breakdown: rho = " << rho_new << " at iteration " << k << std::endl;
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            double* r_temp = axpy(-1.0, matvec(A, x.data()), b, n);
            double* r_tilde_temp = new double[n];
            apply_preconditioner(M.data(), r_temp, n, r_tilde_temp);
            double r_norm = norm(r_tilde_temp, n);
            delete[] r_temp; delete[] r_tilde_temp;
            return {x_out, r_norm, k, residual_history};
        }
        double beta = (rho_new / rho) * (alpha / omega);
        // p = r_tilde + beta (p - omega v)
        std::vector<double> temp(n);
        for (int i = 0; i < n; ++i) temp[i] = p[i] - omega * v[i];
        for (int i = 0; i < n; ++i) p[i] = r_tilde[i] + beta * temp[i];
        // v = M^{-1} A p
        double* Ap = matvec(A, p.data());
        v.assign(n, 0.0);
        apply_preconditioner(M.data(), Ap, n, v.data());
        delete[] Ap;
        double rhat_v = dot(r_hat.data(), v.data(), n);
        if (std::abs(rhat_v) < 1e-10) {
            std::cerr << "Breakdown: r_hat^T v = " << rhat_v << " at iteration " << k << std::endl;
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            double* r_temp = axpy(-1.0, matvec(A, x.data()), b, n);
            double* r_tilde_temp = new double[n];
            apply_preconditioner(M.data(), r_temp, n, r_tilde_temp);
            double r_norm = norm(r_tilde_temp, n);
            delete[] r_temp; delete[] r_tilde_temp;
            return {x_out, r_norm, k, residual_history};
        }
        alpha = rho_new / rhat_v;
        // s = r - alpha A p
        std::vector<double> s(n);
        double* Ap_temp = matvec(A, p.data());
        for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * Ap_temp[i];
        delete[] Ap_temp;
        // Check preconditioned residual: s_tilde = M^{-1} s
        std::vector<double> s_tilde(n);
        apply_preconditioner(M.data(), s.data(), n, s_tilde.data());
        double s_norm = norm(s_tilde.data(), n);
        residual_history.push_back(s_norm);
        if (s_norm < tol_abs) {
            for (int i = 0; i < n; ++i) x[i] += alpha * p[i];
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            return {x_out, s_norm, k + 1, residual_history};
        }
        // t = M^{-1} A s_tilde
        double* As_tilde = matvec(A, s_tilde.data());
        std::vector<double> t(n);
        apply_preconditioner(M.data(), As_tilde, n, t.data());
        delete[] As_tilde;
        double t_t = dot(t.data(), t.data(), n);
        if (std::abs(t_t) < 1e-10) {
            std::cerr << "Breakdown: t^T t = " << t_t << " at iteration " << k << std::endl;
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            double* r_temp = axpy(-1.0, matvec(A, x.data()), b, n);
            double* r_tilde_temp = new double[n];
            apply_preconditioner(M.data(), r_temp, n, r_tilde_temp);
            double r_norm = norm(r_tilde_temp, n);
            delete[] r_temp; delete[] r_tilde_temp;
            return {x_out, r_norm, k, residual_history};
        }
        omega = dot(t.data(), s_tilde.data(), n) / t_t;
        // x = x + alpha p + omega s_tilde
        for (int i = 0; i < n; ++i) x[i] += alpha * p[i] + omega * s_tilde[i];
        // r = s - omega A s_tilde
        double* As_tilde_temp = matvec(A, s_tilde.data());
        for (int i = 0; i < n; ++i) r[i] = s[i] - omega * As_tilde_temp[i];
        delete[] As_tilde_temp;
        // Check preconditioned residual: r_tilde = M^{-1} r
        apply_preconditioner(M.data(), r.data(), n, r_tilde.data());
        double r_norm = norm(r_tilde.data(), n);
        if (r_norm < 0) {
            std::cerr << "Error: Residual became NaN or Inf at iteration " << k + 1 << std::endl;
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            return {x_out, r_norm, k + 1, residual_history};
        }
        if (k % 100 == 0) {
            std::cout << "Iteration " << k << ": Preconditioned Residual = " << r_norm << std::endl;
        }
        if (r_norm < tol_abs) {
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            return {x_out, r_norm, k + 1, residual_history};
        }
        rho = rho_new;
    }

    double* r_temp = axpy(-1.0, matvec(A, x.data()), b, n);
    double* r_tilde_temp = new double[n];
    apply_preconditioner(M.data(), r_temp, n, r_tilde_temp);
    double r_norm = norm(r_tilde_temp, n);
    delete[] r_temp; delete[] r_tilde_temp;
    double* x_out = new double[n];
    for (int i = 0; i < n; ++i) x_out[i] = x[i];
    return {x_out, r_norm, k + 1, residual_history};
}

double* generate_rhs(const CSRMatrix& A) {
    std::vector<double> x_true(A.n, 1.0); // x_true = [1, 1, ..., 1]
    double* b = matvec(A, x_true.data());
    std::cout << "Generated b = A * x_true, where x_true = [1, 1, ..., 1]" << std::endl;
    return b;
}

void write_solution(const double* x, int n, const std::string& filename, const std::vector<double>& residual_history) {
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
}

int main() {
    std::string filename = "../data/suitesparse/1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) {
        free_csr_matrix(A);
        return 1;
    }

    double* b = generate_rhs(A);
    std::vector<double> x_true(A.n, 1.0); // x_true = [1, 1, ..., 1]

    auto start = std::chrono::high_resolution_clock::now();
    Result result = bicgstab(A, b, 2 * A.n, 1e-8);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final preconditioned residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    double* Ax = matvec(A, result.x);
    double* r_temp = axpy(-1.0, Ax, b, A.n);
    double* M = new double[A.n];
    compute_diagonal_preconditioner(A, M);
    double* r_tilde = new double[A.n];
    apply_preconditioner(M, r_temp, A.n, r_tilde);
    double verify_residual = norm(r_tilde, A.n);
    std::cout << "Verification preconditioned residual: " << verify_residual << std::endl;
    delete[] Ax; delete[] r_temp; delete[] M; delete[] r_tilde;
    double* error = axpy(-1.0, x_true.data(), result.x, A.n);
    double error_norm = norm(error, A.n);
    std::cout << "Error norm vs x_true: " << error_norm << std::endl;
    write_solution(result.x, A.n, "bicgstab_solution.csv", result.residual_history);

    delete[] b;
    delete[] result.x;
    free_csr_matrix(A);
    return 0;
}