#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>

struct CSRMatrix {
    int n;
    __PROMISE__* values;
    int* col_indices;
    int* row_ptr;
    int nnz; 
};

bool compare_by_column(const std::pair<int, __PROMISE__>& a, const std::pair<int, __PROMISE__>& b) {
    return a.first < b.first;
}

struct Entry { int row, col; __PROMISE__ val; };

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

    Entry* entries = new Entry[2 * nz];
    int entry_count = 0;

    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            delete[] entries;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        __PROMISE__ val;
        ss >> i >> j >> val;
        if (i < 1 || j < 1 || i > n || j > n) {
            std::cerr << "Error: Invalid indices in Matrix Market file" << std::endl;
            delete[] entries;
            return A;
        }
        i--; j--;
        entries[entry_count++] = {i, j, val};
        if (i != j) entries[entry_count++] = {j, i, val};
    }

    int* nnz_per_row = new int[n]();
    for (int k = 0; k < entry_count; ++k) {
        nnz_per_row[entries[k].row]++;
    }

    A.nnz = entry_count;
    A.values = new __PROMISE__[entry_count];
    A.col_indices = new int[entry_count];
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row[i];
    }

    std::sort(entries, entries + entry_count,
        [](const Entry& a, const Entry& b) {
            return a.row == b.row ? a.col < b.col : a.row < b.row;
        });

    for (int k = 0; k < entry_count; ++k) {
        A.col_indices[k] = entries[k].col;
        A.values[k] = entries[k].val;
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << entry_count << " non-zeros" << std::endl;

    delete[] entries;
    delete[] nnz_per_row;
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

__PROMISE__* matvec(const CSRMatrix& A, const __PROMISE__* x) {
    __PROMISE__* y = new __PROMISE__[A.n]();
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return y;
}

__PROMISE__ dot(const __PROMISE__* a, const __PROMISE__* b, int n) {
    __PROMISE__ sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

__PROMISE__* axpy(__PROMISE__ alpha, const __PROMISE__* x, const __PROMISE__* y, int n) {
    __PROMISE__* result = new __PROMISE__[n];
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

__PROMISE__ norm(const __PROMISE__* v, int n) {
    __PROMISE__ d = dot(v, v, n);
    if (isnan(d) || isinf(d)) return -1.0;
    return sqrt(d);
}

void compute_diagonal_preconditioner(const CSRMatrix& A, __PROMISE__* M) {
    bool has_zero_diagonal = false;
    __PROMISE__ min_diag = std::numeric_limits<__PROMISE__>::max();
    __PROMISE__ max_diag = 0.0;
    for (int i = 0; i < A.n; ++i) {
        M[i] = 0.0;
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

void apply_preconditioner(const __PROMISE__* M, const __PROMISE__* r, int n, __PROMISE__* z) {
    for (int i = 0; i < n; ++i) {
        z[i] = M[i] * r[i];
    }
}

struct Result {
    __PROMISE__* x;
    __PROMISE__ residual;
    int iterations;
    __PROMISE__* residual_history;
    size_t residual_history_size;
};

Result bicgstab(const CSRMatrix& A, const __PROMISE__* b, int max_iter = 1000, __PROMISE__ tol = 1e-12) {
    int n = A.n;
    __PROMISE__* x = new __PROMISE__[n]();
    __PROMISE__* r = new __PROMISE__[n];
    for (int i = 0; i < n; ++i) r[i] = b[i]; // r = b - Ax (x = 0)
    __PROMISE__* r_hat = new __PROMISE__[n];
    __PROMISE__* p = new __PROMISE__[n];
    __PROMISE__* z = new __PROMISE__[n];
    __PROMISE__* M = new __PROMISE__[n];
    compute_diagonal_preconditioner(A, M);
    apply_preconditioner(M, r, n, r_hat); // r_hat = M^{-1} r_0
    for (int i = 0; i < n; ++i) p[i] = r_hat[i]; // p_0 = M^{-1} r_0
    __PROMISE__* v = new __PROMISE__[n]();
    __PROMISE__* residual_history = new __PROMISE__[max_iter]; // Allocate max possible size
    size_t residual_history_size = 0;

    __PROMISE__ rho = 1.0, alpha = 1.0, omega = 1.0;
    __PROMISE__ initial_norm = norm(b, n);
    if (initial_norm < 0) {
        std::cerr << "Error: Initial b has invalid norm" << std::endl;
        __PROMISE__* x_out = new __PROMISE__[n];
        for (int i = 0; i < n; ++i) x_out[i] = x[i];
        delete[] x; delete[] r; delete[] r_hat; delete[] p; delete[] z; delete[] M; delete[] v;
        return {x_out, -1.0, 0, residual_history, 0};
    }
    std::cout << "Initial norm of b: " << initial_norm << std::endl;
    double tol_abs = tol;

    int k;
    for (k = 0; k < max_iter; ++k) {
        // Compute preconditioned residual: r_tilde = M^{-1} r
        __PROMISE__* r_tilde = new __PROMISE__[n];
        apply_preconditioner(M, r, n, r_tilde);
        __PROMISE__ rho_new = dot(r_hat, r_tilde, n);
        if (abs(rho_new) < 1e-10) {
            std::cerr << "Breakdown: rho = " << rho_new << " at iteration " << k << std::endl;
            __PROMISE__* x_out = new __PROMISE__[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            __PROMISE__* r_temp = axpy(-1.0, matvec(A, x), b, n);
            __PROMISE__* r_tilde_temp = new __PROMISE__[n];
            apply_preconditioner(M, r_temp, n, r_tilde_temp);
            __PROMISE__ r_norm = norm(r_tilde_temp, n);
            delete[] r_temp; delete[] r_tilde_temp; delete[] r_tilde;
            delete[] x; delete[] r; delete[] r_hat; delete[] p; delete[] z; delete[] M; delete[] v;
            return {x_out, r_norm, k, residual_history, residual_history_size};
        }
        __PROMISE__ beta = (rho_new / rho) * (alpha / omega);
        // p = r_tilde + beta (p - omega v)
        __PROMISE__* temp = new __PROMISE__[n];
        for (int i = 0; i < n; ++i) temp[i] = p[i] - omega * v[i];
        for (int i = 0; i < n; ++i) p[i] = r_tilde[i] + beta * temp[i];
        delete[] temp;
        // v = M^{-1} A p
        __PROMISE__* Ap = matvec(A, p);
        for (int i = 0; i < n; ++i) v[i] = 0.0;
        apply_preconditioner(M, Ap, n, v);
        delete[] Ap;
        __PROMISE__ rhat_v = dot(r_hat, v, n);
        if (abs(rhat_v) < 1e-10) {
            std::cerr << "Breakdown: r_hat^T v = " << rhat_v << " at iteration " << k << std::endl;
            __PROMISE__* x_out = new __PROMISE__[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            __PROMISE__* r_temp = axpy(-1.0, matvec(A, x), b, n);
            __PROMISE__* r_tilde_temp = new __PROMISE__[n];
            apply_preconditioner(M, r_temp, n, r_tilde_temp);
            __PROMISE__ r_norm = norm(r_tilde_temp, n);
            delete[] r_temp; delete[] r_tilde_temp; delete[] r_tilde;
            delete[] x; delete[] r; delete[] r_hat; delete[] p; delete[] z; delete[] M; delete[] v;
            return {x_out, r_norm, k, residual_history, residual_history_size};
        }
        alpha = rho_new / rhat_v;
        // s = r - alpha A p
        __PROMISE__* s = new __PROMISE__[n];
        __PROMISE__* Ap_temp = matvec(A, p);
        for (int i = 0; i < n; ++i) s[i] = r[i] - alpha * Ap_temp[i];
        delete[] Ap_temp;
        // Check preconditioned residual: s_tilde = M^{-1} s
        __PROMISE__* s_tilde = new __PROMISE__[n];
        apply_preconditioner(M, s, n, s_tilde);
        __PROMISE__ s_norm = norm(s_tilde, n);
        residual_history[residual_history_size++] = s_norm;
        if (s_norm < tol_abs) {
            for (int i = 0; i < n; ++i) x[i] += alpha * p[i];
            __PROMISE__* x_out = new __PROMISE__[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            delete[] r; delete[] r_hat; delete[] p; delete[] z; delete[] M; delete[] v;
            delete[] r_tilde; delete[] s; delete[] s_tilde; delete[] x;
            return {x_out, s_norm, k + 1, residual_history, residual_history_size};
        }
        // t = M^{-1} A s_tilde
        __PROMISE__* As_tilde = matvec(A, s_tilde);
        __PROMISE__* t = new __PROMISE__[n];
        apply_preconditioner(M, As_tilde, n, t);
        delete[] As_tilde;
        __PROMISE__ t_t = dot(t, t, n);
        if (abs(t_t) < 1e-10) {
            std::cerr << "Breakdown: t^T t = " << t_t << " at iteration " << k << std::endl;
            __PROMISE__* x_out = new __PROMISE__[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            __PROMISE__* r_temp = axpy(-1.0, matvec(A, x), b, n);
            __PROMISE__* r_tilde_temp = new __PROMISE__[n];
            apply_preconditioner(M, r_temp, n, r_tilde_temp);
            __PROMISE__ r_norm = norm(r_tilde_temp, n);
            delete[] r_temp; delete[] r_tilde_temp; delete[] r_tilde; delete[] s; delete[] s_tilde; delete[] t;
            delete[] x; delete[] r; delete[] r_hat; delete[] p; delete[] z; delete[] M; delete[] v;
            return {x_out, r_norm, k, residual_history, residual_history_size};
        }
        omega = dot(t, s_tilde, n) / t_t;
        // x = x + alpha p + omega s_tilde
        for (int i = 0; i < n; ++i) x[i] += alpha * p[i] + omega * s_tilde[i];
        // r = s - omega A s_tilde
        __PROMISE__* As_tilde_temp = matvec(A, s_tilde);
        for (int i = 0; i < n; ++i) r[i] = s[i] - omega * As_tilde_temp[i];
        delete[] As_tilde_temp;
        // Check preconditioned residual: r_tilde = M^{-1} r
        apply_preconditioner(M, r, n, r_tilde);
        __PROMISE__ r_norm = norm(r_tilde, n);
        delete[] r_tilde; delete[] s; delete[] s_tilde; delete[] t;
        if (r_norm < 0) {
            std::cerr << "Error: Residual became NaN or Inf at iteration " << k + 1 << std::endl;
            __PROMISE__* x_out = new __PROMISE__[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            delete[] x; delete[] r; delete[] r_hat; delete[] p; delete[] z; delete[] M; delete[] v;
            return {x_out, r_norm, k + 1, residual_history, residual_history_size};
        }
        if (k % 100 == 0) {
            std::cout << "Iteration " << k << ": Preconditioned Residual = " << r_norm << std::endl;
        }
        if (r_norm < tol_abs) {
            __PROMISE__* x_out = new __PROMISE__[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            delete[] x; delete[] r; delete[] r_hat; delete[] p; delete[] z; delete[] M; delete[] v;
            return {x_out, r_norm, k + 1, residual_history, residual_history_size};
        }
        rho = rho_new;
    }

    __PROMISE__* r_temp = axpy(-1.0, matvec(A, x), b, n);
    __PROMISE__* r_tilde_temp = new __PROMISE__[n];
    apply_preconditioner(M, r_temp, n, r_tilde_temp);
    __PROMISE__ r_norm = norm(r_tilde_temp, n);
    delete[] r_temp; delete[] r_tilde_temp;
    __PROMISE__* x_out = new __PROMISE__[n];
    for (int i = 0; i < n; ++i) x_out[i] = x[i];
    delete[] x; delete[] r; delete[] r_hat; delete[] p; delete[] z; delete[] M; delete[] v;
    return {x_out, r_norm, k + 1, residual_history, residual_history_size};
}

__PROMISE__* generate_rhs(const CSRMatrix& A) {
    __PROMISE__* x_true = new __PROMISE__[A.n];
    for (int i = 0; i < A.n; ++i) x_true[i] = 1.0; // x_true = [1, 1, ..., 1]
    __PROMISE__* b = matvec(A, x_true);
    delete[] x_true;
    return b;
}


int main() {
    std::string filename = "psmigr_2.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) {
        free_csr_matrix(A);
        return 1;
    }

    __PROMISE__* b = generate_rhs(A);
    __PROMISE__* x_true = new __PROMISE__[A.n];
    for (int i = 0; i < A.n; ++i) x_true[i] = 1.0; 

    auto start = std::chrono::high_resolution_clock::now();
    Result result = bicgstab(A, b, 2 * A.n, 1e-12);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    __PROMISE__* x_out = new __PROMISE__[A.n];
    for (int i = 0; i < A.n; ++i) x_out[i] = result.x[i];
    
    PROMISE_CHECK_ARRAY(x_out, A.n);

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
    double* error = axpy(-1.0, x_true, result.x, A.n);
    double error_norm = norm(error, A.n);
    std::cout << "Error norm vs x_true: " << error_norm << std::endl;

    delete[] b;
    delete[] x_true;
    delete[] result.x;
    delete[] result.residual_history;
    free_csr_matrix(A);
    return 0;
}