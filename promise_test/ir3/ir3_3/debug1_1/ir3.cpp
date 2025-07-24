#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>

template<class T> struct Matrix {
    T** data;
    int rows, cols;
};

template<class T> struct Vector {
    T* data;
    int size;
};

template<class T> struct CSRMatrix {
    int n = 0;
    T* values = nullptr;
    int* col_indices = nullptr;
    int* row_ptr = nullptr;
    int nnz = 0;
};

template<class T> struct Entry {
    int row, col;
    T val;
};

template<typename T> void create_matrix(Matrix<T>& mat, int rows, int cols) {
    mat.rows = rows;
    mat.cols = cols;
    mat.data = new T*[rows];
    for (int i = 0; i < rows; ++i) {
        mat.data[i] = new T[cols]();
    }
}

template<class T> void free_matrix(Matrix<T>& mat) {
    for (int i = 0; i < mat.rows; ++i) {
        delete[] mat.data[i];
    }
    delete[] mat.data;
    mat.data = nullptr;
    mat.rows = 0;
    mat.cols = 0;
}

template<class T> Vector<T> create_vector(int size) {
    Vector<T> vec;
    vec.size = size;
    vec.data = new T[size]();
    return vec;
}

template<class T> void free_vector(Vector<T>& vec) {
    delete[] vec.data;
    vec.data = nullptr;
    vec.size = 0;
}

template<class T> CSRMatrix<T> read_mtx_file(const std::string& filename) {
    CSRMatrix<T> A;
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

    Entry<T>* entries = new Entry<T>[2 * nz];
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
        float val;
        ss >> i >> j >> val;
        if (i < 1 || j < 1 || i > n || j > n) {
            std::cerr << "Error: Invalid indices in Matrix Market file" << std::endl;
            delete[] entries;
            return A;
        }
        i--; j--;
        entries[entry_count++] = {i, j, static_cast<T>(val)};
        if (i != j) entries[entry_count++] = {j, i, static_cast<T>(val)};
    }

    int* nnz_per_row = new int[n]();
    for (int k = 0; k < entry_count; ++k) {
        nnz_per_row[entries[k].row]++;
    }

    A.nnz = entry_count;
    A.values = new T[A.nnz];
    A.col_indices = new int[A.nnz];
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row[i];
    }

    std::sort(entries, entries + entry_count,
        [](const Entry<T>& a, const Entry<T>& b) {
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

template<class T> void free_csr_matrix(CSRMatrix<T>& A) {
    delete[] A.values;
    delete[] A.col_indices;
    delete[] A.row_ptr;
    A.values = nullptr;
    A.col_indices = nullptr;
    A.row_ptr = nullptr;
    A.n = 0;
    A.nnz = 0;
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

template<class T> void matvec(const CSRMatrix<T>& A, const T* x, T* y) {
    for (int i = 0; i < A.n; ++i) {
        y[i] = 0.0;
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            y[i] += A.values[k] * x[A.col_indices[k]];
        }
    }
}

template<class T> Matrix<T> csr_to_dense(const CSRMatrix<T>& A) {
    Matrix<T> dense;
    create_matrix(dense, A.n, A.n);
    for (int i = 0; i < A.n; ++i) {
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            dense.data[i][A.col_indices[k]] = A.values[k];
        }
    }
    return dense;
}

template<class T1, class T2> void lu_factorization(const Matrix<double>& A, Matrix<T1>& L, Matrix<T2>& U, int* P) {
    int n = A.rows;
    create_matrix(L, n, n);
    create_matrix(U, n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            U.data[i][j] = static_cast<T2>(A.data[i][j]);
        }
        P[i] = i;
        L.data[i][i] = 1.0;
    }

    for (int k = 0; k < n; ++k) {
        float max_val = abs(U.data[k][k]);
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

template<typename T1, typename T2> Vector<T2> forward_substitution(const Matrix<T1>& L, const Vector<T2>& b, const int* P) {
    int n = L.rows;
    Vector<T2> y = create_vector<T2>(n);
    for (int i = 0; i < n; ++i) {
        T2 sum = 0.0;
        for (int j = 0; j < i; ++j) {
            sum += L.data[i][j] * y.data[j];
        }
        y.data[i] = b.data[P[i]] - sum;
    }
    return y;
}

template<typename T1, typename T2> Vector<T2> backward_substitution(const Matrix<T1>& U, const Vector<T2>& y) {
    int n = U.rows;
    Vector<T2> x = create_vector<T2>(n);
    for (int i = n - 1; i >= 0; --i) {
        T2 sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += U.data[i][j] * x.data[j];
        }
        x.data[i] = (y.data[i] - sum) / U.data[i][i];
    }
    return x;
}

template<class T> Vector<T> vec_sub(const Vector<T>& a, const Vector<T>& b) {
    Vector<T> result = create_vector<T>(a.size);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] - b.data[i];
    }
    return result;
}

template<class T> Vector<T> vec_add(const Vector<T>& a, const Vector<T>& b) {
    Vector<T> result = create_vector<T>(a.size);
    for (int i = 0; i < a.size; ++i) {
        result.data[i] = a.data[i] + b.data[i];
    }
    return result;
}

template<class T> Vector<T> round_to_low_prec(const Vector<T>& x) {
    Vector<T> result = create_vector<T>(x.size);
    for (int i = 0; i < x.size; ++i) {
        result.data[i] = static_cast<flx::floatx<4, 3>>(x.data[i]);
    }
    return result;
}


template<class T> Vector<T> iterative_refinement(const CSRMatrix<T>& A_csr, const Vector<T>& b, int max_iter, float tol, double*& residual_history, int& history_size) {
    if (A_csr.n > 10000) {
        std::cerr << "Error: Matrix too large for dense conversion\n";
        return create_vector<T>(0);
    }

    history_size = 0;
    residual_history = new double[max_iter];

    Matrix<double> A = csr_to_dense<double>(A_csr);
    Matrix<double> L;
    Matrix<double> U;
    
    int* P = new int[A_csr.n];
    try {
        lu_factorization(A, L, U, P);
    } catch (const std::exception& e) {
        std::cerr << "LU factorization failed: " << e.what() << "\n";
        delete[] P;
        free_matrix(A);
        return create_vector<T>(0);
    }

    Vector<T> y = forward_substitution<double, T>(L, b, P);
    Vector<T> x = backward_substitution<double, T>(U, y);
    free_vector(y);

    Vector<T> Ax = create_vector<T>(A_csr.n);
    Vector<T> r = create_vector<T>(A_csr.n);
    Vector<T> d = create_vector<T>(A_csr.n);

    for (int iter = 0; iter < max_iter; ++iter) {
        matvec(A_csr, x.data, Ax.data);
        r = vec_sub(b, Ax);

        float norm_r = 0.0;
        for (int i = 0; i < r.size; ++i) {
            norm_r += r.data[i] * r.data[i];
        }
        norm_r = sqrt(norm_r);
        residual_history[history_size++] = norm_r;

        Vector<T> r_low = round_to_low_prec(r);
        Vector<T> y_d = forward_substitution<double, T>(L, r_low, P);
        d = backward_substitution<double, T>(U, y_d);
        free_vector(y_d);
        free_vector(r_low);

        Vector<T> x_new = vec_add(x, d);
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
        CSRMatrix<double> A = read_mtx_file<double>(filename);
        if (A.n == 0) {
            std::cerr << "Failed to load matrix\n";
            return 1;
        }

        Vector<double> b = create_vector<double>(A.n);
        double* b_raw = generate_rhs(A.n);
        for (int i = 0; i < A.n; ++i) {
            b.data[i] = b_raw[i];
        }
        delete[] b_raw;

        double* residual_history = nullptr;
        int history_size = 0;
        Vector<double> x = iterative_refinement<double>(A, b, 1000, 1e-8, residual_history, history_size);

        if (x.size == 0) {
            std::cerr << "Failed to solve system\n";
            free_csr_matrix(A);
            free_vector(b);
            delete[] residual_history;
            return 1;
        }
        

        double check_solution[A.n];
        for (int i = 0; i < A.n; ++i) {
            check_solution[i] = x.data[i];
        }

        PROMISE_CHECK_ARRAY(check_solution, A.n);
        

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