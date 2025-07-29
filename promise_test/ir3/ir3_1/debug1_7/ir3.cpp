#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>

double* create_dense_matrix(int rows, int cols) {
    return new double[rows * cols]();
}

void free_dense_matrix(double* mat) {
    delete[] mat;
}

double* create_vector(int size) {
    return new double[size]();
}

void free_vector(double* vec) {
    delete[] vec;
}

struct Entry {
    int row, col;
    double val;
};

void read_mtx_file(std::string& filename, double*& values, int*& col_indices, int*& row_ptr, int& n, int& nnz) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line) && line[0] == '%') {}

    std::stringstream ss(line);
    int m, nz;
    ss >> n >> m >> nz;
    if (n != m) {
        std::cerr << "Error: Matrix must be square" << std::endl;
        return;
    }

    Entry* entries = new Entry[2 * nz];
    int entry_count = 0;

    for (int k = 0; k < nz; ++k) {
        if (!std::getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            delete[] entries;
            return;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        double val;
        ss >> i >> j >> val;
        if (i < 1 || j < 1 || i > n || j > n) {
            std::cerr << "Error: Invalid indices in Matrix Market file" << std::endl;
            delete[] entries;
            return;
        }
        i--; j--;
        entries[entry_count++] = {i, j, val};
        if (i != j) entries[entry_count++] = {j, i, val};
    }

    int* nnz_per_row = new int[n]();
    for (int k = 0; k < entry_count; ++k) {
        nnz_per_row[entries[k].row]++;
    }

    nnz = entry_count;
    values = new double[nnz];
    col_indices = new int[nnz];
    row_ptr = new int[n + 1];
    row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        row_ptr[i + 1] = row_ptr[i] + nnz_per_row[i];
    }

    std::sort(entries, entries + entry_count,
        [](Entry& a, Entry& b) {
            return a.row == b.row ? a.col < b.col : a.row < b.row;
        });

    for (int k = 0; k < nnz; ++k) {
        col_indices[k] = entries[k].col;
        values[k] = entries[k].val;
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << nnz << " non-zeros" << std::endl;

    delete[] nnz_per_row;
    delete[] entries;
}

void free_csr_matrix(double*& values, int*& col_indices, int*& row_ptr) {
    delete[] values;
    delete[] col_indices;
    delete[] row_ptr;
    values = NULL;
    col_indices = NULL;
    row_ptr = NULL;
}

double* generate_rhs(int n) {
    double* b = new double[n];
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dis(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

void matvec(double* values, int* col_indices, int* row_ptr, int n, double* x, double* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            y[i] += values[k] * x[col_indices[k]];
        }
    }
}

double* csr_to_dense(double* values, int* col_indices, int* row_ptr, int n, int nnz) {
    double* dense = create_dense_matrix(n, n);
    for (int i = 0; i < n; ++i) {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            dense[i * n + col_indices[k]] = values[k];
        }
    }
    return dense;
}

void lu_factorization(double* A, int n, double* L, double* U, int* P) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U[i * n + j] = A[i * n + j];
        }
        P[i] = i;
        L[i * n + i] = 1.0;
    }

    for (int k = 0; k < n; ++k) {
        float max_val = abs(U[k * n + k]);
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (abs(U[i * n + k]) > max_val) {
                max_val = abs(U[i * n + k]);
                pivot = i;
            }
        }
        if (abs(max_val) < 1e-15) {
            throw std::runtime_error("Matrix singular or nearly singular");
        }
        if (pivot != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(U[k * n + j], U[pivot * n + j]);
                if (j < k) {
                    std::swap(L[k * n + j], L[pivot * n + j]);
                }
            }
            std::swap(P[k], P[pivot]);
        }
        for (int i = k + 1; i < n; ++i) {
            L[i * n + k] = U[i * n + k] / U[k * n + k];
            for (int j = k; j < n; ++j) {
                U[i * n + j] -= L[i * n + k] * U[k * n + j];
            }
        }
    }
}

double* forward_substitution(double* L, int n, double* b, int* P) {
    double* y = create_vector(n);
    for (int i = 0; i < n; ++i) {
        float sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = b[P[i]] - sum;
    }
    return y;
}

double* backward_substitution(double* U, int n, double* y) {
    double* x = create_vector(n);
    for (int i = n - 1; i >= 0; --i) {
        float sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += U[i * n + j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i * n + i];
    }
    return x;
}

double* vec_sub(double* a, double* b, int size) {
    double* result = create_vector(size);
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

double* vec_add(double* a, double* b, int size) {
    double* result = create_vector(size);
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

double* initial_solve(double* L, double* U, int n, int* P, double* b) {
    double* y = forward_substitution(L, n, b, P);
    double* x = backward_substitution(U, n, y);
    free_vector(y);
    return x;
}

double* compute_residual(double* values, int* col_indices, int* row_ptr, int n, double* b, double* x) {
    double* Ax = create_vector(n);
    matvec(values, col_indices, row_ptr, n, x, Ax);
    double* r = vec_sub(b, Ax, n);
    free_vector(Ax);
    return r;
}

double* solve_correction(double* L, double* U, int n, int* P, double* r) {
    double* y = forward_substitution(L, n, r, P);
    double* d = backward_substitution(U, n, y);
    free_vector(y);
    return d;
}

double* update_solution(double* x, double* d, int size) {
    return vec_add(x, d, size);
}

void write_solution(double* x, int size, std::string& filename, double* residual_history, int history_size) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }
    file << "x\n";
    for (int i = 0; i < size; ++i) {
        file << x[i] << "\n";
    }
    file << "\nResidual History\n";
    for (int i = 0; i < history_size; ++i) {
        file << i << "," << residual_history[i] << "\n";
    }
    file.close();
}

double* iterative_refinement(double* values, int* col_indices, int* row_ptr, int n, int nnz, double* b, int max_iter, float tol, double*& residual_history, int& history_size) {
    if (n > 10000) {
        std::cerr << "Error: Matrix too large for dense conversion\n";
        return NULL;
    }

    history_size = 0;
    residual_history = new double[max_iter];

    double* A = csr_to_dense(values, col_indices, row_ptr, n, nnz);
    double* L = create_dense_matrix(n, n);
    double* U = create_dense_matrix(n, n);
    
    int* P = new int[n];
    try {
        lu_factorization(A, n, L, U, P);
    } catch (std::exception& e) {
        std::cerr << "LU factorization failed: " << e.what() << "\n";
        delete[] P;
        free_dense_matrix(A);
        free_dense_matrix(L);
        free_dense_matrix(U);
        return NULL;
    }

    free_dense_matrix(A);

    double* x = initial_solve(L, U, n, P, b);

    for (int iter = 0; iter < max_iter; ++iter) {
        double* r = compute_residual(values, col_indices, row_ptr, n, b, x);

        float norm_r = 0.0;
        for (int i = 0; i < n; ++i) {
            norm_r += r[i] * r[i];
        }
        norm_r = sqrt(norm_r);
        residual_history[history_size++] = norm_r;

        if (norm_r < tol) {
            std::cout << "Converged after " << iter + 1 << " iterations\n";
            free_vector(r);
            break;
        }

        double* d = solve_correction(L, U, n, P, r);
        free_vector(r);

        double* x_new = update_solution(x, d, n);
        free_vector(d);
        free_vector(x);
        x = x_new;
    }

    free_dense_matrix(L);
    free_dense_matrix(U);
    delete[] P;
    return x;
}

int main() {
    std::string filename = "1138_bus.mtx";

    try {
        double* values = NULL;
        int* col_indices = NULL;
        int* row_ptr = NULL;
        int n = 0;
        int nnz = 0;
        read_mtx_file(filename, values, col_indices, row_ptr, n, nnz);
        if (n == 0) {
            std::cerr << "Failed to load matrix\n";
            return 1;
        }

        double* b = generate_rhs(n);

        double* residual_history = NULL;
        int history_size = 0;
        double* x = iterative_refinement(values, col_indices, row_ptr, n, nnz, b, 1000, 1e-8, residual_history, history_size);


        PROMISE_CHECK_ARRAY(x, n);
        if (x == NULL) {
            std::cerr << "Failed to solve system\n";
            free_csr_matrix(values, col_indices, row_ptr);
            free_vector(b);
            delete[] residual_history;
            return 1;
        }
        

        std::string output_file = "solution.txt";
        write_solution(x, n, output_file, residual_history, history_size);

        std::cout << "\nResidual History:\n";
        for (int i = 0; i < history_size; ++i) {
            std::cout << "Iteration " << i << ": " << residual_history[i] << "\n";
        }

        free_csr_matrix(values, col_indices, row_ptr);
        free_vector(b);
        free_vector(x);
        delete[] residual_history;

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}