#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>

struct CSRMatrix {
    int n;
    int nnz;
    __PROMISE__* values;
    int* col_indices;
    int* row_ptr;

    CSRMatrix() : n(0), nnz(0), values(nullptr), col_indices(nullptr), row_ptr(nullptr) {}
    ~CSRMatrix() { delete[] values; delete[] col_indices; delete[] row_ptr; }
};

CSRMatrix read_mtx_file(const std::string& filename) {
    CSRMatrix A;
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

    struct RowEntry {
        int* cols;
        __PROMISE__* vals;
        int size;
        int capacity;
    };
    RowEntry* temp = new RowEntry[n];
    for (int i = 0; i < n; ++i) {
        temp[i].size = 0;
        temp[i].capacity = 10;
        temp[i].cols = new int[temp[i].capacity];
        temp[i].vals = new __PROMISE__[temp[i].capacity];
    }

    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            for (int i = 0; i < n; ++i) {
                delete[] temp[i].cols;
                delete[] temp[i].vals;
            }
            delete[] temp;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        __PROMISE__ val;
        ss >> i >> j >> val;
        i--; j--;

        if (temp[i].size == temp[i].capacity) {
            temp[i].capacity *= 2;
            int* new_cols = new int[temp[i].capacity];
            __PROMISE__* new_vals = new __PROMISE__[temp[i].capacity];
            for (int p = 0; p < temp[i].size; ++p) {
                new_cols[p] = temp[i].cols[p];
                new_vals[p] = temp[i].vals[p];
            }
            delete[] temp[i].cols;
            delete[] temp[i].vals;
            temp[i].cols = new_cols;
            temp[i].vals = new_vals;
        }
        temp[i].cols[temp[i].size] = j;
        temp[i].vals[temp[i].size] = val;
        temp[i].size++;

        if (i != j) {
            if (temp[j].size == temp[j].capacity) {
                temp[j].capacity *= 2;
                int* new_cols = new int[temp[j].capacity];
                __PROMISE__* new_vals = new __PROMISE__[temp[j].capacity];
                for (int p = 0; p < temp[j].size; ++p) {
                    new_cols[p] = temp[j].cols[p];
                    new_vals[p] = temp[j].vals[p];
                }
                delete[] temp[j].cols;
                delete[] temp[j].vals;
                temp[j].cols = new_cols;
                temp[j].vals = new_vals;
            }
            temp[j].cols[temp[j].size] = i;
            temp[j].vals[temp[j].size] = val;
            temp[j].size++;
        }
    }

    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    A.nnz = 0;
    for (int i = 0; i < n; ++i) {
        A.nnz += temp[i].size;
        A.row_ptr[i + 1] = A.row_ptr[i] + temp[i].size;
    }

    A.values = new __PROMISE__[A.nnz];
    A.col_indices = new int[A.nnz];
    int pos = 0;
    for (int i = 0; i < n; ++i) {
        for (int p = 0; p < temp[i].size - 1; ++p) {
            for (int q = p + 1; q < temp[i].size; ++q) {
                if (temp[i].cols[p] > temp[i].cols[q]) {
                    int temp_col = temp[i].cols[p];
                    __PROMISE__ temp_val = temp[i].vals[p];
                    temp[i].cols[p] = temp[i].cols[q];
                    temp[i].vals[p] = temp[i].vals[q];
                    temp[i].cols[q] = temp_col;
                    temp[i].vals[q] = temp_val;
                }
            }
        }
        for (int p = 0; p < temp[i].size; ++p) {
            A.col_indices[pos] = temp[i].cols[p];
            A.values[pos] = temp[i].vals[p];
            pos++;
        }
    }

    for (int i = 0; i < n; ++i) {
        delete[] temp[i].cols;
        delete[] temp[i].vals;
    }
    delete[] temp;

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << A.nnz << " non-zeros" << std::endl;
    return A;
}

__PROMISE__ norm(const __PR_vec__* v, int n) {
    __PROMISE__ sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += v[i] * v[i];
    }
    return sqrt(sum);
}

void jacobi(const CSRMatrix& A, const __PR_vec__* b, __PR_vec__* x, int max_iter = 100, __PR_scalar__ tol = 1e-2) {
    int n = A.n;
    __PR_vec__* x_new = new __PR_vec__[n]();
    __PR_vec__* residual = new __PR_vec__[n];

    for (int iter = 0; iter < max_iter; ++iter) {
        for (int i = 0; i < n; ++i) {
            __PR_scalar__ sum = 0.0;
            __PR_scalar__ diag = 0.0;
            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
                if (A.col_indices[j] == i) {
                    diag = A.values[j];
                } else {
                    sum += A.values[j] * x[A.col_indices[j]];
                }
            }
            x_new[i] = (b[i] - sum) / diag;
        }

        for (int i = 0; i < n; ++i) {
            residual[i] = x_new[i] - x[i];
            x[i] = x_new[i];
        }

        __PROMISE__ res_norm = norm(residual, n);
        if (iter % 10 == 0) {
            std::cout << "Iteration " << iter << ": residual = " << res_norm << std::endl;
        }
        if (res_norm < tol) {
            break;
        }
    }

    delete[] x_new;
    delete[] residual;
}

__PR_vec__* generate_rhs(int n) {
    __PR_vec__* b = new __PR_vec__[n];
    // std::random_device rd;
    std::mt19937 gen(223);
    std::uniform_real_distribution<> dis(0, 1.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

int main() {
    std::string filename = "rdb5000.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) return 1;

    __PR_vec__* b = generate_rhs(A.n);
    __PR_vec__* x = new __PR_vec__[A.n]();
    __PR_vec__* x_ref = new __PR_vec__[A.n]();

    // Reference solution (double)
    for (int i = 0; i < A.n; ++i) {
        x_ref[i] = 0.0;
    }
    jacobi(A, b, x_ref, 100, 1e-2);

    auto start = std::chrono::high_resolution_clock::now();
    jacobi(A, b, x, 100, 1e-2);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    __PROMISE__ error = norm(x, A.n);
    for (int i = 0; i < A.n; ++i) {
        __PROMISE__ diff = x[i] - x_ref[i];
        error += diff * diff;
    }
    error = sqrt(error);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Computation time: " << duration.count() << " ms" << std::endl;
    std::cout << "Error: " << error << std::endl;

    PROMISE_CHECK_VAR(error);

    delete[] b;
    delete[] x;
    delete[] x_ref;

    return 0;
}