#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>

struct CSRMatrix {
    int n;
    double* values;
    int* col_indices;
    int* row_ptr;
    int nnz; // Number of non-zeros
};

bool compare_by_column(const std::pair<int, __PROMISE__>& a, const std::pair<int, __PROMISE__>& b) {
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
    while (getline(file, line) && line[0] == '%') {} // Skip comments

    std::stringstream ss(line);
    int n, m, nz;
    if (!(ss >> n >> m >> nz)) {
        std::cerr << "Error: Invalid header format in " << filename << std::endl;
        return A;
    }
    if (n != m) {
        std::cerr << "Error: Matrix must be square" << std::endl;
        return A;
    }
    A.n = n;

    // Temporary storage for entries
    struct Entry { int row, col; __PROMISE__ val; };
    Entry* entries = new Entry[2 * nz]; // Max possible entries (symmetric)
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
        if (!(ss >> i >> j >> val)) {
            std::cerr << "Error: Invalid entry format at line " << k + 1 << std::endl;
            delete[] entries;
            return A;
        }
        i--; j--; // Convert to 0-based indexing
        if (i < 0 || i >= n || j < 0 || j >= n) {
            std::cerr << "Error: Invalid indices at line " << k + 1 << std::endl;
            delete[] entries;
            return A;
        }
        entries[entry_count++] = {i, j, val};
        if (i != j) entries[entry_count++] = {j, i, val}; // Symmetric matrix
    }

    // Count non-zeros per row
    int* nnz_per_row = new int[n]();
    for (int k = 0; k < entry_count; ++k) {
        nnz_per_row[entries[k].row]++;
    }

    // Allocate CSR arrays
    A.nnz = entry_count;
    A.values = new __PROMISE__[entry_count];
    A.col_indices = new int[entry_count];
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row[i];
    }

    // Sort entries by row and column
    std::sort(entries, entries + entry_count, 
        [](const Entry& a, const Entry& b) { 
            return a.row == b.row ? a.col < b.col : a.row < b.row; 
        });

    // Fill CSR arrays
    for (int k = 0; k < entry_count; ++k) {
        A.col_indices[k] = entries[k].col;
        A.values[k] = entries[k].val;
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << entry_count << " non-zeros" << std::endl;

    // Clean up
    delete[] nnz_per_row;
    delete[] entries;
    return A;
}

void free_csr_matrix(CSRMatrix& A) {
    delete[] A.values;
    delete[] A.col_indices;
    delete[] A.row_ptr;
    A.values = nullptr;
    A.col_indices = nullptr;
    A.row_ptr = nullptr;
    A.n = 0;
    A.nnz = 0;
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

void axpy(__PROMISE__ alpha, const __PROMISE__* x, const __PROMISE__* y, int n, __PROMISE__* result) {
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
}

__PROMISE__ norm(const __PROMISE__* v, int n) {
    return sqrt(abs(dot(v, v, n)));
}

void compute_diagonal_preconditioner(const CSRMatrix& A, __PROMISE__* M) {
    bool has_zero_diagonal = false;
    for (int i = 0; i < A.n; ++i) {
        M[i] = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] == i) {
                M[i] = A.values[j];
                break;
            }
        }
        if (abs(M[i]) < 1e-10) {
            has_zero_diagonal = true;
            M[i] = 1.0; // Fallback for zero diagonal
        } else {
            M[i] = 1.0 / M[i];
        }
    }
    if (has_zero_diagonal) {
        std::cerr << "Warning: Matrix has zero or near-zero diagonal elements, which may affect convergence" << std::endl;
    }
}

void apply_preconditioner(const __PROMISE__* M, const __PROMISE__* r, int n, __PROMISE__* z) {
    for (int i = 0; i < n; ++i) {
        z[i] = M[i] * r[i];
    }
}

struct Solution {
    __PROMISE__* x;
    __PROMISE__ residual;
    int iterations;
};

Solution conjugate_gradient(const CSRMatrix& A, const __PROMISE__* b, int max_iter = 1000, __PROMISE__ tol = 1e-12) {
    int n = A.n;
    Solution result = {new __PROMISE__[n](), 0.0, 0};

    __PROMISE__* r = new __PROMISE__[n];
    __PROMISE__* p = new __PROMISE__[n];
    __PROMISE__* z = new __PROMISE__[n];
    __PROMISE__* Ap = new __PROMISE__[n];
    __PROMISE__* M = new __PROMISE__[n];

    // Initialize residual
    for (int i = 0; i < n; ++i) {
        r[i] = b[i];
        result.x[i] = 0.0; // Explicitly initialize x to zero
    }

    __PROMISE__ b_norm = norm(b, n);
    if (b_norm < 1e-16) b_norm = 1e-16;

    compute_diagonal_preconditioner(A, M);
    apply_preconditioner(M, r, n, z);
    for (int i = 0; i < n; ++i) p[i] = z[i];
    __PROMISE__ rz_old = dot(r, z, n);
    __PROMISE__ tol2 = tol * tol * b_norm * b_norm;

    __PROMISE__ initial_rz = rz_old;
    __PROMISE__ prev_rz = rz_old;
    int stagnant_count = 0;
    const __PROMISE__ eps = 1e-16;

    std::cout << "Initial residual norm: " << norm(r, n) / b_norm << std::endl;

    for (int k = 0; k < max_iter; ++k) {
        // Compute Ap = A*p
        delete[] Ap; // Free previous Ap
        Ap = matvec(A, p);
        __PROMISE__ pAp = dot(p, Ap, n);
        if (abs(pAp) < eps) {
            std::cerr << "Error: pAp too small at iteration " << k + 1 << ": " << pAp << std::endl;
            result.iterations = k + 1;
            break;
        }
        __PROMISE__ alpha = rz_old / pAp;

        // Update x: x = x + alpha*p
        axpy(alpha, p, result.x, n, result.x);

        // Update r: r = r - alpha*Ap
        axpy(-alpha, Ap, r, n, r);

        // Check residual
        __PROMISE__ rel_residual = norm(r, n) / b_norm;
        if (k % 100 == 0) {
            std::cout << "Iteration " << k + 1 << ": Relative residual = " << rel_residual << std::endl;
        }

        // Apply preconditioner: z = M*r
        apply_preconditioner(M, r, n, z);
        __PROMISE__ rz_new = dot(r, z, n);

        // Check for divergence or numerical issues
        if (rz_new > 1e10 * initial_rz || isnan(rz_new) || isinf(rz_new)) {
            std::cerr << "Error: Divergence detected at iteration " << k + 1 << ": rz = " << rz_new << std::endl;
            result.iterations = k + 1;
            break;
        }

        // Check for stagnation
        if (abs(rz_new - prev_rz) < eps * rz_new && k > 0) {
            stagnant_count++;
            if (stagnant_count > 5) {
                std::cerr << "Warning: Stagnation detected at iteration " << k + 1 << std::endl;
                result.iterations = k + 1;
                break;
            }
        } else {
            stagnant_count = 0;
        }

        // Check convergence
        if (rel_residual < tol) {
            result.iterations = k + 1;
            break;
        }

        // Update p: p = z + beta*p
        __PROMISE__ beta = rz_new / rz_old;
        axpy(beta, p, z, n, p);
        rz_old = rz_new;
        prev_rz = rz_new;
    }

    result.residual = norm(r, n);
    delete[] r;
    delete[] p;
    delete[] z;
    delete[] Ap;
    delete[] M;
    return result;
}

__PROMISE__* generate_rhs(int n) {
    __PROMISE__* b = new __PROMISE__[n];
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0, 1.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

int main() {
    std::string filename = "1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) {
        std::cerr << "Failed to load matrix" << std::endl;
        free_csr_matrix(A);
        return 1;
    }

    __PROMISE__* b = generate_rhs(A.n);

    auto start = std::chrono::high_resolution_clock::now();
    Solution result = conjugate_gradient(A, b, 2 * A.n, 1e-12);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<__PROMISE__, std::milli> duration = end - start;

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    // Verify the solution
    __PROMISE__* Ax = matvec(A, result.x);
    __PROMISE__* temp = new __PROMISE__[A.n];
    axpy(-1.0, Ax, b, A.n, temp);
    __PROMISE__ verify_residual = norm(temp, A.n);

    double check_result[A.n];
    for (int i = 0; i < A.n; i++) {
        check_result[i] = result.x[i];
    }

    PROMISE_CHECK_ARRAY(check_result, A.n);
    std::cout << "Verification residual: " << verify_residual << std::endl;
    delete[] Ax;
    delete[] temp;
    delete[] b;
    delete[] result.x;
    free_csr_matrix(A);
    return 0;
}