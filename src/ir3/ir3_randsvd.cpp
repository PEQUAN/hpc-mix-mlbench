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

template<typename To, typename From>
To* convert_vector(From* src, int size) {
    To* dest = create_vector<To>(size);
    for (int i = 0; i < size; ++i) {
        dest[i] = static_cast<To>(src[i]);
    }
    return dest;
}

template<typename T>
T* generate_random_orthogonal(int n, unsigned int seed_offset = 0) {
    T* mat = create_dense_matrix<T>(n, n);
    std::mt19937 gen(42 + seed_offset);
    std::normal_distribution<T> dis(0.0, 1.0);
    for(int i = 0; i < n * n; ++i) {
        mat[i] = dis(gen);
    }
    // Modified Gram-Schmidt
    for(int j = 0; j < n; ++j) {
        for(int k = 0; k < j; ++k) {
            T dot = 0.0;
            for(int i = 0; i < n; ++i) {
                dot += mat[i * n + j] * mat[i * n + k];
            }
            for(int i = 0; i < n; ++i) {
                mat[i * n + j] -= dot * mat[i * n + k];
            }
        }
        T norm = 0.0;
        for(int i = 0; i < n; ++i) {
            norm += mat[i * n + j] * mat[i * n + j];
        }
        norm = std::sqrt(norm);
        if(norm != 0.0) {
            for(int i = 0; i < n; ++i) {
                mat[i * n + j] /= norm;
            }
        }
    }
    return mat;
}

template<typename T>
void dense_to_csr(T* dense, int n, T*& values, int*& col_indices, int*& row_ptr, int& nnz) {
    nnz = n * n;
    values = new T[nnz];
    col_indices = new int[nnz];
    row_ptr = new int[n + 1];
    row_ptr[0] = 0;
    int idx = 0;
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            values[idx] = dense[i * n + j];
            col_indices[idx] = j;
            ++idx;
        }
        row_ptr[i + 1] = idx;
    }
}

template<typename T>
void generate_randsvd(T*& values, int*& col_indices, int*& row_ptr, int& n, int& nnz, T kappa, int mode = 3) {
    // Implementing mode 3: geometrically distributed singular values, sigma_max=1, sigma_min=1/kappa
    T* U = generate_random_orthogonal<T>(n, 0);
    T* V = generate_random_orthogonal<T>(n, 1); // Different seed
    T* sigma = new T[n];
    sigma[0] = 1.0;
    if (n > 1) {
        T exponent = -1.0 / (n - 1.0);
        T r = std::pow(kappa, exponent);
        for (int i = 1; i < n; ++i) {
            sigma[i] = sigma[i - 1] * r;
        }
    }
    T* A = create_dense_matrix<T>(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            T sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += U[i * n + k] * sigma[k] * V[j * n + k]; // V^T[k][j] = V[j][k]
            }
            A[i * n + j] = sum;
        }
    }
    free_dense_matrix<T>(U);
    free_dense_matrix<T>(V);
    delete[] sigma;
    // Convert to CSR
    dense_to_csr<T>(A, n, values, col_indices, row_ptr, nnz);
    free_dense_matrix<T>(A);
    std::cout << "Generated matrix: " << n << " x " << n << " with " << nnz << " non-zeros (dense)" << std::endl;
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

template<typename ValT, typename XT, typename YT>
void matvec(ValT* values, int* col_indices, int* row_ptr, int n, XT* x, YT* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = YT(0);
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            y[i] += static_cast<YT>(values[k]) * static_cast<YT>(x[col_indices[k]]);
        }
    }
}

template<typename ToType, typename FromType>
ToType* csr_to_dense(FromType* values, int* col_indices, int* row_ptr, int n, int nnz) {
    ToType* dense = create_dense_matrix<ToType>(n, n);
    for (int i = 0; i < n; ++i) {
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            dense[i * n + col_indices[k]] = static_cast<ToType>(values[k]);
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

template<typename MatT, typename VecT>
VecT* forward_substitution(MatT* L, int n, VecT* b, int* P) {
    VecT* y = create_vector<VecT>(n);
    for (int i = 0; i < n; ++i) {
        VecT sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += static_cast<VecT>(L[i * n + j]) * y[j];
        }
        y[i] = b[P[i]] - sum;
    }
    return y;
}

template<typename MatT, typename VecT>
VecT* backward_substitution(MatT* U, int n, VecT* y) {
    VecT* x = create_vector<VecT>(n);
    for (int i = n - 1; i >= 0; --i) {
        VecT sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += static_cast<VecT>(U[i * n + j]) * x[j];
        }
        x[i] = (y[i] - sum) / static_cast<VecT>(U[i * n + i]);
    }
    return x;
}

template<typename T>
T* vec_sub(T* a, T* b, int size) {
    T* result = create_vector<T>(size);
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

template<typename T, typename U>
T* vec_add(T* a, U* b, int size) {
    T* result = create_vector<T>(size);
    for (int i = 0; i < size; ++i) {
        result[i] = a[i] + static_cast<T>(b[i]);
    }
    return result;
}

template<typename Policy>
typename Policy::SubType* initial_solve(typename Policy::LUType* L, typename Policy::LUType* U, int n, int* P, typename Policy::SubType* b) {
    using MatT = typename Policy::LUType;
    using VecT = typename Policy::SubType;
    VecT* y = forward_substitution<MatT, VecT>(L, n, b, P);
    VecT* x = backward_substitution<MatT, VecT>(U, n, y);
    free_vector<VecT>(y);
    return x;
}

template<typename Policy>
typename Policy::ResType* compute_residual(typename Policy::ResType* values, int* col_indices, int* row_ptr, int n, typename Policy::ResType* b, typename Policy::UpdateType* x) {
    typename Policy::ResType* Ax = create_vector<typename Policy::ResType>(n);
    matvec<typename Policy::ResType, typename Policy::UpdateType, typename Policy::ResType>(values, col_indices, row_ptr, n, x, Ax);
    typename Policy::ResType* r = vec_sub<typename Policy::ResType>(b, Ax, n);
    free_vector<typename Policy::ResType>(Ax);
    return r;
}

template<typename Policy>
typename Policy::CorrType* solve_correction(typename Policy::LUType* L, typename Policy::LUType* U, int n, int* P, typename Policy::CorrType* r) {
    using MatT = typename Policy::LUType;
    using VecT = typename Policy::CorrType;
    VecT* y = forward_substitution<MatT, VecT>(L, n, r, P);
    VecT* d = backward_substitution<MatT, VecT>(U, n, y);
    free_vector<VecT>(y);
    return d;
}

template<typename Policy>
typename Policy::UpdateType* update_solution(typename Policy::UpdateType* x, typename Policy::CorrType* d, int size) {
    return vec_add<typename Policy::UpdateType, typename Policy::CorrType>(x, d, size);
}

template<typename XT, typename RT>
void write_solution(XT* x, int size, std::string& filename, RT* residual_history, int history_size) {
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

    typename Policy::LUType* A = csr_to_dense<typename Policy::LUType, typename Policy::ResType>(values, col_indices, row_ptr, n, nnz);
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

    typename Policy::SubType* b_sub = convert_vector<typename Policy::SubType, typename Policy::ResType>(b, n);
    typename Policy::UpdateType* x = initial_solve<Policy>(L, U, n, P, b_sub);
    free_vector<typename Policy::SubType>(b_sub);

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

        typename Policy::CorrType* r_corr = convert_vector<typename Policy::CorrType, typename Policy::ResType>(r, n);
        free_vector<typename Policy::ResType>(r);
        typename Policy::CorrType* d = solve_correction<Policy>(L, U, n, P, r_corr);
        free_vector<typename Policy::CorrType>(r_corr);

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
    try {
        // Define precision policy (user can modify these types)
        using MyPolicy = PrecisionPolicy<
            float,          // LU factorization precision
            double,         // Substitution precision
            long double,    // Residual computation precision
            double,         // Correction solve precision
            double          // Solution update precision
        >;

        MyPolicy::ResType tolerance = 1e-8; // User-defined tolerance
        int n = 100; // User-defined matrix size
        MyPolicy::ResType kappa = 1e6; // User-defined condition number

        MyPolicy::ResType* values = nullptr;
        int* col_indices = nullptr;
        int* row_ptr = nullptr;
        int nnz = 0;
        generate_randsvd<MyPolicy::ResType>(values, col_indices, row_ptr, n, nnz, kappa);
        if (n == 0) {
            std::cerr << "Failed to generate matrix\n";
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
        write_solution<MyPolicy::UpdateType, MyPolicy::ResType>(x, n, output_file, residual_history, history_size);

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