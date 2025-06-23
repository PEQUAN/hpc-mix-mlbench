#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>

struct CSRMatrix {
    int n = 0;          // Matrix dimension
    __PROMISE__* values = nullptr;    // Non-zero values
    int* col_indices = nullptr;  // Column indices of non-zeros
    int* row_ptr = nullptr;      // Row pointers
    int nnz = 0;        // Number of non-zeros
};

struct Entry {
    int row, col;
    __PROMISE__ val;
};

CSRMatrix read_mtx_file(const std::string& filename) {
    CSRMatrix A;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return A;
    }

    std::string line;
    while (std::getline(file, line) && line[0] == '%') {}

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
        if (!std::getline(file, line)) {
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
    A.values = new __PROMISE__[A.nnz];
    A.col_indices = new int[A.nnz];
    A.row_ptr = new int[n + 1];
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

__PROMISE__* generate_rhs(const CSRMatrix& A) {
    __PROMISE__* x_true = new __PROMISE__[A.n];
    __PROMISE__* b = new __PROMISE__[A.n]();
    for (int i = 0; i < A.n; ++i) {
        x_true[i] = 1.0;
    }
    for (int i = 0; i < A.n; ++i) {
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            b[i] += A.values[k] * x_true[A.col_indices[k]];
        }
    }
    std::cout << "Generated b = A * x_true, where x_true = [1, 1, ..., 1]" << std::endl;
    delete[] x_true;
    return b;
}

void matvec(const CSRMatrix& A, const __PROMISE__* x, __PROMISE__* y) {
    for (int i = 0; i < A.n; ++i) {
        y[i] = 0.0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            y[i] += A.values[k] * x[A.col_indices[k]];
        }
    }
}

struct Matrix {
    __PROMISE__** data;
    int rows, cols;
};

struct Vector {
    __PROMISE__* data;
    int size;
};


Matrix create_matrix(int rows, int cols) {
    Matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = new __PROMISE__*[rows];
    for (int i = 0; i < rows; ++i) {
        mat.data[i] = new __PROMISE__[cols]();
    }
    return mat;
}


void free_matrix(Matrix& mat) {
    for (int i = 0; i < mat.rows; ++i) {
        delete[] mat.data[i];
    }
    delete[] mat.data;
    mat.data = nullptr;
    mat.rows = 0;
    mat.cols = 0;
}


Matrix csr_to_dense(const CSRMatrix& A) {
    Matrix dense = create_matrix(A.n, A.n);
    for (int i = 0; i < A.n; ++i) {
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            dense.data[i][A.col_indices[k]] = A.values[k];
        }
    }
    return dense;
}

Vector create_vector(int size) {
    Vector vec;
    vec.size = size;
    vec.data = new __PROMISE__[size]();
    return vec;
}

void free_vector(Vector& vec) {
    delete[] vec.data;
    vec.data = nullptr;
    vec.size = 0;
}


void lu_factorization(const Matrix& A, Matrix& L, Matrix& U, int* P) {
    int n = A.rows;
// Perform LU decomposition with partial pivoting: PA = LU
    L = create_matrix(n, n);
    U = create_matrix(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U.data[i][j] = A.data[i][j];
        }
        P[i] = i;
        L.data[i][i] = 1.0;
    }

    for (int k = 0; k < n; ++k) {
        __PROMISE__ max_val = abs(U.data[k][k]);
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (abs(U.data[i][k]) > max_val) {
                max_val = abs(U.data[i][k]);
                pivot = i;
            }
        }
        if (abs(max_val) < 1e-15) {
            throw std::runtime_error("Matrix singular or nearly singular");
        }
        if (pivot != k) {
            std::swap(U.data[k], U.data[pivot]);
            std::swap(P[k], P[pivot]);
            for (int j = 0; j < k; ++j) {
                std::swap(L.data[k][j], L.data[pivot][j]);
            }
        }
        for (int i = k + 1; i < n; ++i) {
            L.data[i][k] = U.data[i][k] / U.data[k][k];
            for (int j = k; j < n; ++j) {
                U.data[i][j] -= L.data[i][k] * U.data[k][j];
            }
        }
    }
}


Vector forward_substitution(const Matrix& L, const Vector& b, const int* P) {
    int n = L.rows;// Solve Ly = Pb (forward substitution)
    Vector y = create_vector(n);
    for (int i = 0; i < n; ++i) {
        __PROMISE__ sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L.data[i][j] * y.data[j];
        }
        y.data[i] = b.data[P[i]] - sum;
    }
    return y;
}


Vector backward_substitution(const Matrix& U, const Vector& y) {
    int n = U.rows;// Solve Ux = y (backward substitution)
    Vector x = create_vector(n);
    for (int i = n - 1; i >= 0; --i) {
        __PROMISE__ sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += U.data[i][j] * x.data[j];
        }
        x.data[i] = (y.data[i] - sum) / U.data[i][i];
    }
    return x;
}

// Vector subtraction
Vector vec_sub(const Vector& a, const Vector& b) {
    Vector result = create_vector(a.size);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] - b.data[i];
    }
    return result;
}


Vector vec_add(const Vector& a, const Vector& b) {
    Vector result = create_vector(a.size);// Vector addition
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
}


Vector round_to_low_prec(const Vector& x) {
    Vector result = create_vector(x.size);// Round vector to single precision
    for (int i = 0; i < x.size; ++i) {
        result.data[i] = static_cast<__PROMISE__>(x.data[i]);
    }
    return result;
}




Vector iterative_refinement(const CSRMatrix& A_csr, const Vector& b, int max_iter, __PROMISE__ tol, __PROMISE__*& residual_history, int& history_size) {
    if (A_csr.n > 10000) {// Solve Ax = b using iterative refinement
        std::cerr << "Error: Matrix too large for dense conversion\n";
        return create_vector(0);
    }

    history_size = 0;
    residual_history = new __PROMISE__[max_iter];

    Matrix A = csr_to_dense(A_csr);
    Matrix L, U;
    int* P = new int[A_csr.n];
    try {
        lu_factorization(A, L, U, P);
    } catch (const std::exception& e) {
        std::cerr << "LU factorization failed: " << e.what() << "\n";
        delete[] P;
        free_matrix(A);
        return create_vector(0);
    }

    Vector y = forward_substitution(L, b, P);
    Vector x = backward_substitution(U, y);
    free_vector(y);

    Vector Ax = create_vector(A_csr.n);
    Vector r = create_vector(A_csr.n);
    Vector d = create_vector(A_csr.n);

    for (int iter = 0; iter < max_iter; ++iter) {
        matvec(A_csr, x.data, Ax.data);
        r = vec_sub(b, Ax);

        __PROMISE__ norm_r = 0.0;
        for (int i = 0; i < r.size; ++i) {
            norm_r += r.data[i] * r.data[i];
        }
        norm_r = sqrt(norm_r);
        residual_history[history_size++] = norm_r;

        Vector r_low = round_to_low_prec(r);
        Vector y_d = forward_substitution(L, r_low, P);
        d = backward_substitution(U, y_d);
        free_vector(y_d);
        free_vector(r_low);

        Vector x_new = vec_add(x, d);
        free_vector(x);
        x = x_new;

        if (norm_r < tol) {
            std::cout << "Converged after " << iter + 1 << " iterations\n";
            break;
        }
    }

    free_matrix(A);
    free_matrix(L);
    free_matrix(U);
    delete[] P;
    free_vector(Ax);
    free_vector(r);
    free_vector(d);
    return x;
}


int main() {
    std::string filename = "1138_bus.mtx";

    try {
        CSRMatrix A = read_mtx_file(filename);
        if (A.n == 0) {
            std::cerr << "Failed to load matrix\n";
            return 1;
        }

        Vector b = create_vector(A.n);
        __PROMISE__* b_raw = generate_rhs(A);
        for (int i = 0; i < A.n; ++i) {
            b.data[i] = b_raw[i];
        }
        delete[] b_raw;

        __PROMISE__* residual_history = nullptr;
        int history_size = 0;
        Vector x = iterative_refinement(A, b, 10, 1e-6, residual_history, history_size);

        if (x.size == 0) {
            std::cerr << "Failed to solve system\n";
            free_csr_matrix(A);
            free_vector(b);
            delete[] residual_history;
            return 1;
        }

        double solution_check[x.size];
        for (int i = 0; i < x.size; ++i) {
            solution_check[i] = x.data[i];
        }

        PROMISE_CHECK_ARRAY(solution_check, x.size);

        std::cout << "\nResidual History:\n";
        for (int i = 0; i < history_size; ++i) {
            std::cout << "Iteration " << i << ": " << residual_history[i] << "\n";
        }

        free_csr_matrix(A);
        free_vector(b);
        free_vector(x);
        delete[] residual_history;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}