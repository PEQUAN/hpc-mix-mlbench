#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>
#include <type_traits>

// Precision policy to control precision for different steps
template<typename LUPrecision, typename SubPrecision, typename ResPrecision, typename CorrPrecision, typename UpdatePrecision>
struct PrecisionPolicy {
    using LUType = LUPrecision;
    using SubType = SubPrecision;
    using ResType = ResPrecision;
    using CorrType = CorrPrecision;
    using UpdateType = UpdatePrecision;
};

template<typename T>
T* create_dense_matrix(int rows, int cols) {
    return new T[rows * cols]();
}

template<typename T>
void free_dense_matrix(T* mat) {
    delete[] mat;
}

template<typename T>
T* create_vector(int size) {
    return new T[size]();
}

template<typename T>
void free_vector(T* vec) {
    delete[] vec;
}

struct Entry {
    int row, col;
    float val;
};

template<typename T>
void read_mtx_file(std::string& filename, T*& values, int*& col_indices, int*& row_ptr, int& n, int& nnz) {
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
        float val;
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
    values = new T[nnz];
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
        values[k] = static_cast<T>(entries[k].val);
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << nnz << " non-zeros" << std::endl;

    delete[] nnz_per_row;
    delete[] entries;
}

template<typename T>
void free_csr_matrix(T*& values, int*& col_indices, int*& row_ptr) {
    delete[] values;
    delete[] col_indices;
    delete[] row_ptr;
    values = nullptr;
    col_indices = nullptr;
    row_ptr = nullptr;
}

template<typename T>
T* generate_rhs(int n) {
    T* b = new T[n];
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dis(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

template<typename T>
void matvec(T* values, int* col_indices, int* row_ptr, int n, T* x, T* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = 0.0;
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            y[i] += values[k] * x[col_indices[k]];
        }
    }
}

template<typename T>
T* csr_to_dense(T* values, int* col_indices, int* row_ptr, int n, int nnz) {
    T* dense = create_dense_matrix<T>(n, n);
    for (int i = 0; i < n; ++i) {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            dense[i * n + col_indices[k]] = values[k];
        }
    }
    return dense;
}

template<typename T>
void lu_factorization(T* A, int n, T* L, T* U, int* P) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U[i * n + j] = A[i * n + j];
        }
        P[i] = i;
        L[i * n + i] = 1.0;
    }

    for (int k = 0; k < n; ++k) {
        T max_val = std::abs(U[k * n + k]);
        int pivot = k;
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(U[i * n + k]) > max_val) {
                max_val = std::abs(U[i * n + k]);
                pivot = i;
            }
        }
        if (std::abs(max_val) < std::numeric_limits<T>::epsilon()) {
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

template<typename T>
T* forward_substitution(T* L, int n, T* b, int* P) {
    T* y = create_vector<T>(n);
    for (int i = 0; i < n; ++i) {
        T sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L[i * n + j] * y[j];
        }
        y[i] = b[P[i]] - sum;
    }
    return y;
}

template<typename T>
T* backward_substitution(T* U, int n, T* y) {
    T* x = create_vector<T>(n);
    for (int i = n - 1; i >= 0; --i) {
        T sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += U[i * n + j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i * n + i];
    }
    return x;
}

template<typename T, typename U>
T* vec_sub(T* a, U* b, int size) {
    T* result = create_vector<T>(size);
    for (int i = 0; i < size; ++i) {
        result[i] = static_cast<T>(a[i] - b[i]);
    }
    return result;
}

template<typename T, typename U>
T* vec_add(T* a, U* b, int size) {
    T* result = create_vector<T>(size);
    for (int i = 0; i < size; ++i) {
        result[i] = static_cast<T>(a[i] + b[i]);
    }
    return result;
}

template<typename Policy>
typename Policy::SubType* initial_solve(typename Policy::LUType* L, typename Policy::LUType* U, int n, int* P, typename Policy::SubType* b) {
    typename Policy::SubType* y = forward_substitution<typename Policy::SubType>(L, n, b, P);
    typename Policy::SubType* x = backward_substitution<typename Policy::SubType>(U, n, y);
    free_vector<typename Policy::SubType>(y);
    return x;
}

template<typename Policy>
typename Policy::ResType* compute_residual(typename Policy::ResType* values, int* col_indices, int* row_ptr, int n, typename Policy::ResType* b, typename Policy::UpdateType* x) {
    typename Policy::ResType* Ax = create_vector<typename Policy::ResType>(n);
    matvec<typename Policy::ResType>(values, col_indices, row_ptr, n, x, Ax);
    typename Policy::ResType* r = vec_sub<typename Policy::ResType, typename Policy::ResType>(b, Ax, n);
    free_vector<typename Policy::ResType>(Ax);
    return r;
}

template<typename Policy>
typename Policy::CorrType* solve_correction(typename Policy::LUType* L, typename Policy::LUType* U, int n, int* P, typename Policy::CorrType* r) {
    typename Policy::CorrType* y = forward_substitution<typename Policy::CorrType>(L, n, r, P);
    typename Policy::CorrType* d = backward_substitution<typename Policy::CorrType>(U, n, y);
    free_vector<typename Policy::CorrType>(y);
    return d;
}

template<typename Policy>
typename Policy::UpdateType* update_solution(typename Policy::UpdateType* x, typename Policy::CorrType* d, int size) {
    return vec_add<typename Policy::UpdateType, typename Policy::CorrType>(x, d, size);
}

template<typename T>
void write_solution(T* x, int size, std::string& filename, T* residual_history, int history_size) {
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

template<typename Policy>
typename Policy::UpdateType* iterative_refinement(typename Policy::ResType* values, int* col_indices, int* row_ptr, int n, int nnz, typename Policy::ResType* b, int max_iter, typename Policy::ResType tol, typename Policy::ResType*& residual_history, int& history_size) {
    if (n > 10000) {
        std::cerr << "Error: Matrix too large for dense conversion\n";
        return nullptr;
    }

    history_size = 0;
    residual_history = new typename Policy::ResType[max_iter];

    typename Policy::LUType* A = csr_to_dense<typename Policy::LUType>(values, col_indices, row_ptr, n, nnz);
    typename Policy::LUType* L = create_dense_matrix<typename Policy::LUType>(n, n);
    typename Policy::LUType* U = create_dense_matrix<typename Policy::LUType>(n, n);
    
    int* P = new int[n];
    try {
        lu_factorization<typename Policy::LUType>(A, n, L, U, P);
    } catch (std::exception& e) {
        std::cerr << "LU factorization failed: " << e.what() << "\n";
        delete[] P;
        free_dense_matrix<typename Policy::LUType>(A);
        free_dense_matrix<typename Policy::LUType>(L);
        free_dense_matrix<typename Policy::LUType>(U);
        return nullptr;
    }

    free_dense_matrix<typename Policy::LUType>(A);

    typename Policy::UpdateType* x = initial_solve<Policy>(L, U, n, P, b);

    for (int iter = 0; iter < max_iter; ++iter) {
        typename Policy::ResType* r = compute_residual<Policy>(values, col_indices, row_ptr, n, b, x);

        typename Policy::ResType norm_r = 0.0;
        for (int i = 0; i < n; ++i) {
            norm_r += r[i] * r[i];
        }
        norm_r = std::sqrt(norm_r);
        residual_history[history_size++] = norm_r;

        if (norm_r < tol) {
            std::cout << "Converged after " << iter + 1 << " iterations\n";
            free_vector<typename Policy::ResType>(r);
            break;
        }

        typename Policy::CorrType* d = solve_correction<Policy>(L, U, n, P, r);
        free_vector<typename Policy::ResType>(r);

        typename Policy::UpdateType* x_new = update_solution<Policy>(x, d, n);
        free_vector<typename Policy::CorrType>(d);
        free_vector<typename Policy::UpdateType>(x);
        x = x_new;
    }

    free_dense_matrix<typename Policy::LUType>(L);
    free_dense_matrix<typename Policy::LUType>(U);
    delete[] P;
    return x;
}

int main() {
    std::string filename = "1138_bus.mtx";

    try {
        // Define precision policy (user can modify these types)
        using MyPolicy = PrecisionPolicy<
            float,          // LU factorization precision
            float,         // Substitution precision
            float,    // Residual computation precision
            float,         // Correction solve precision
            float          // Solution update precision
        >;

        MyPolicy::ResType tolerance = 1e-8; // User-defined tolerance

        MyPolicy::ResType* values = nullptr;
        int* col_indices = nullptr;
        int* row_ptr = nullptr;
        int n = 0;
        int nnz = 0;
        read_mtx_file<MyPolicy::ResType>(filename, values, col_indices, row_ptr, n, nnz);
        if (n == 0) {
            std::cerr << "Failed to load matrix\n";
            return 1;
        }

        MyPolicy::ResType* b = generate_rhs<MyPolicy::ResType>(n);

        MyPolicy::ResType* residual_history = nullptr;
        int history_size = 0;
        MyPolicy::UpdateType* x = iterative_refinement<MyPolicy>(values, col_indices, row_ptr, n, nnz, b, 1000, tolerance, residual_history, history_size);

        if (x == nullptr) {
            std::cerr << "Failed to solve system\n";
            free_csr_matrix<MyPolicy::ResType>(values, col_indices, row_ptr);
            free_vector<MyPolicy::ResType>(b);
            delete[] residual_history;
            return 1;
        }

        std::string output_file = "solution.txt";
        write_solution<MyPolicy::UpdateType>(x, n, output_file, residual_history, history_size);

        std::cout << "\nResidual History:\n";
        for (int i = 0; i < history_size; ++i) {
            std::cout << "Iteration " << i << ": " << residual_history[i] << "\n";
        }

        free_csr_matrix<MyPolicy::ResType>(values, col_indices, row_ptr);
        free_vector<MyPolicy::ResType>(b);
        free_vector<MyPolicy::UpdateType>(x);
        delete[] residual_history;

    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}