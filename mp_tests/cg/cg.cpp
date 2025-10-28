#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <omp.h>

struct CSRMatrix {
    int n;
    __PROMISE__* values;
    int* col_indices;
    int* row_ptr;
    int nnz; // number of non-zeros
};

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

void matvec(const CSRMatrix& A, const __PROMISE__* x, __PROMISE__* y) {
    #pragma omp parallel for
    for (int i = 0; i < A.n; ++i) {
        y[i] = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
}

__PROMISE__ dot(const __PROMISE__* a, const __PROMISE__* b, int n) {
    __PROMISE__ sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void axpy(__PROMISE__ alpha, const __PROMISE__* x, const __PROMISE__* y, int n, __PROMISE__* result) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
}

__PROMISE__ norm(const __PROMISE__* v, int n) {
    __PROMISE__ d = dot(v, v, n);
    return sqrt(d);
}

void compute_diagonal_preconditioner(const CSRMatrix& A, __PROMISE__* M) {
    bool has_zero_diagonal = false;
    __PROMISE__ min_diag = 9999.9;
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
            M[i] = 1.0;
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
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        z[i] = M[i] * r[i];
    }
}

struct Result {
    __PROMISE__* x;
    __PROMISE__ residual;
    int iterations;
    __PROMISE__* residual_history;
    int residual_history_size;
};

Result pcg(const CSRMatrix& A, const __PROMISE__* b, int max_iter = 1000, __PROMISE__ tol = 1e-12) {
    int n = A.n;
    __PROMISE__* x = new __PROMISE__[n]();
    __PROMISE__* r = new __PROMISE__[n];
    for (int i = 0; i < n; ++i) r[i] = b[i]; // r = b - Ax (x = 0)
    __PROMISE__* z = new __PROMISE__[n];
    __PROMISE__* M = new __PROMISE__[n];
    compute_diagonal_preconditioner(A, M);
    apply_preconditioner(M, r, n, z); // z = M^{-1} r
    __PROMISE__* p = new __PROMISE__[n];
    for (int i = 0; i < n; ++i) p[i] = z[i]; // p_0 = M^{-1} r_0
    __PROMISE__* residual_history = new __PROMISE__[max_iter];

    __PROMISE__ r_z = dot(r, z, n);
    __PROMISE__ initial_norm = norm(b, n);
    if (initial_norm < 0) {
        std::cerr << "Error: Initial b has invalid norm" << std::endl;
        delete[] r; delete[] z; delete[] M; delete[] p; delete[] residual_history;
        return {x, -1.0, 0, nullptr, 0};
    }
    std::cout << "Initial norm of b: " << initial_norm << std::endl;
    __PROMISE__ tol_abs = tol * initial_norm;

    int k;
    for (k = 0; k < max_iter; ++k) {
        __PROMISE__* Ap = new __PROMISE__[n];
        matvec(A, p, Ap);
        __PROMISE__ p_Ap = dot(p, Ap, n);
        if (abs(p_Ap) < 1e-10) {
            std::cerr << "Breakdown: p^T Ap = " << p_Ap << " at iteration " << k << std::endl;
            __PROMISE__* r_temp = new __PROMISE__[n];
            matvec(A, x, r_temp);
            axpy(-1.0, r_temp, b, n, r_temp);
            __PROMISE__* z_temp = new __PROMISE__[n];
            apply_preconditioner(M, r_temp, n, z_temp);
            __PROMISE__ r_norm = norm(z_temp, n);
            delete[] r_temp; delete[] z_temp; delete[] Ap;
            delete[] r; delete[] z; delete[] M; delete[] p;
            return {x, r_norm, k, residual_history, k};
        }
        __PROMISE__ alpha = r_z / p_Ap;
        axpy(alpha, p, x, n, x);
        axpy(-alpha, Ap, r, n, r);
        apply_preconditioner(M, r, n, z);
        __PROMISE__ r_norm = norm(z, n);
        residual_history[k] = r_norm;
        if (r_norm < tol_abs) {
            delete[] Ap; delete[] r; delete[] z; delete[] M; delete[] p;
            return {x, r_norm, k + 1, residual_history, k + 1};
        }
        __PROMISE__ r_z_new = dot(r, z, n);
        if (abs(r_z_new) < 1e-10) {
            std::cerr << "Breakdown: r^T z = " << r_z_new << " at iteration " << k << std::endl;
            __PROMISE__* r_temp = new __PROMISE__[n];
            matvec(A, x, r_temp);
            axpy(-1.0, r_temp, b, n, r_temp);
            __PROMISE__* z_temp = new __PROMISE__[n];
            apply_preconditioner(M, r_temp, n, z_temp);
            __PROMISE__ r_norm = norm(z_temp, n);
            delete[] r_temp; delete[] z_temp; delete[] Ap;
            delete[] r; delete[] z; delete[] M; delete[] p;
            return {x, r_norm, k, residual_history, k};
        }
        __PROMISE__ beta = r_z_new / r_z;
        axpy(beta, p, z, n, p);
        r_z = r_z_new;
        delete[] Ap;
        if (k % 100 == 0) {
            std::cout << "Iteration " << k << ": Preconditioned Residual = " << r_norm << std::endl;
        }
    }

    __PROMISE__* r_temp = new __PROMISE__[n];
    matvec(A, x, r_temp);
    axpy(-1.0, r_temp, b, n, r_temp);
    __PROMISE__* z_temp = new __PROMISE__[n];
    apply_preconditioner(M, r_temp, n, z_temp);
    __PROMISE__ r_norm = norm(z_temp, n);
    delete[] r_temp; delete[] z_temp;
    delete[] r; delete[] z; delete[] M; delete[] p;
    return {x, r_norm, k, residual_history, k};
}


double* generate_rhs(const CSRMatrix& A) {
    double* x_true = new double[A.n];
    for (int i = 0; i < A.n; ++i) x_true[i] = 1.0; // x_true = [1, 1, ..., 1]
    double* b = new double[A.n];
    matvec(A, x_true, b);
    std::cout << "Generated b = A * x_true, where x_true = [1, 1, ..., 1]" << std::endl;
    delete[] x_true;
    return b;
}

int main() {
    std::string filename = "1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) {
        free_csr_matrix(A);
        return 1;
    }

    __PROMISE__* b = generate_rhs(A);

    auto start = std::chrono::high_resolution_clock::now();
    Result result = pcg(A, b, 2 * A.n, 1e-8);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final preconditioned residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    double solution[A.n];
    for (int i=0; i<A.n; i++){
        solution[i] = result.x[i];
    }

    PROMISE_CHECK_ARRAY(solution, A.n);

    double* r_temp = new double[A.n];
    matvec(A, result.x, r_temp);
    axpy(-1.0, r_temp, b, A.n, r_temp);
    double* M = new double[A.n];
    compute_diagonal_preconditioner(A, M);
    double* z_temp = new double[A.n];
    apply_preconditioner(M, r_temp, A.n, z_temp);
    double verify_residual = norm(z_temp, A.n);
    std::cout << "Verification preconditioned residual: " << verify_residual << std::endl;
    delete[] r_temp; delete[] M; delete[] z_temp;

    delete[] b;
    delete[] result.x;
    delete[] result.residual_history;
    free_csr_matrix(A);
    return 0;
}