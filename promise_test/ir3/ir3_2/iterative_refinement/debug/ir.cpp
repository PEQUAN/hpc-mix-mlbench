#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>

#define MAX_N 10000      // Maximum matrix dimension
#define MAX_NZ 1000000   // Maximum number of non-zeros
#define MAX_NZ_PER_ROW 1000 // Maximum non-zeros per row

struct CSRMatrix {
    int n;
    float* values;
    int* col_indices;
    int* row_ptr;
    int nnz; // number of non-zeros
};

struct Pair {
    int first;
    float second;
};

bool compare_by_column(const Pair& a, const Pair& b) {
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
    //    std::cerr << "Error: Matrix must be square" << std::endl;
        return A;
    }

    if (n > MAX_N || nz > MAX_NZ) {
    //    std::cerr << "Error: Matrix size or non-zeros exceed maximum limits" << std::endl;
        return A;
    }
    A.n = n;
    A.nnz = 0;

    A.values = new float[MAX_NZ];
    A.col_indices = new int[MAX_NZ];
    A.row_ptr = new int[n + 1];

    Pair* temp = new Pair[MAX_N * MAX_NZ_PER_ROW];
    int* temp_sizes = new int[n](); 

    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            // std::cerr << "Error: Unexpected end of file" << std::endl;
            delete[] A.values;
            delete[] A.col_indices;
            delete[] A.row_ptr;
            delete[] temp;
            delete[] temp_sizes;
            A.n = 0;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        float val;
        ss >> i >> j >> val;
        i--; j--;
        if (temp_sizes[i] < MAX_NZ_PER_ROW) {
            temp[i * MAX_NZ_PER_ROW + temp_sizes[i]] = {j, val};
            temp_sizes[i]++;
        }
        if (i != j && temp_sizes[j] < MAX_NZ_PER_ROW) {
            temp[j * MAX_NZ_PER_ROW + temp_sizes[j]] = {i, val};
            temp_sizes[j]++;
        }
    }

    A.row_ptr[0] = 0;
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        std::sort(temp + i * MAX_NZ_PER_ROW, temp + i * MAX_NZ_PER_ROW + temp_sizes[i], compare_by_column);
        A.row_ptr[i + 1] = A.row_ptr[i] + temp_sizes[i];
        for (int k = 0; k < temp_sizes[i]; ++k) {
            if (idx >= MAX_NZ) {
                std::cerr << "Error: Exceeded maximum non-zeros" << std::endl;
                delete[] A.values;
                delete[] A.col_indices;
                delete[] A.row_ptr;
                delete[] temp;
                delete[] temp_sizes;
                A.n = 0;
                return A;
            }
            A.col_indices[idx] = temp[i * MAX_NZ_PER_ROW + k].first;
            A.values[idx] = temp[i * MAX_NZ_PER_ROW + k].second;
            idx++;
        }
    }
    A.nnz = idx;

    delete[] temp;
    delete[] temp_sizes;

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << A.nnz << " non-zeros" << std::endl;
    return A;
}

void free_csr_matrix(CSRMatrix& A) {
    delete[] A.values;
    delete[] A.col_indices;
    delete[] A.row_ptr;
    A.n = 0;
    A.nnz = 0;
}

double* csr_to_dense(const CSRMatrix& A) {
    double* dense = new double[A.n * A.n]();
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            dense[i * A.n + A.col_indices[j]] = A.values[j];
        }
    }
    return dense;
}

void lu_factorize_with_pivoting(double* A, int* pivot, int n) {
    for (int i = 0; i < n; ++i) pivot[i] = i;

    for (int k = 0; k < n; ++k) {
        float max_val = abs(A[k * n + k]);
        int max_idx = k;
        for (int i = k + 1; i < n; ++i) {
            if (abs(A[i * n + k]) > max_val) {
                max_val = abs(A[i * n + k]);
                max_idx = i;
            }
        }
        if (max_val < 1e-10) {
            std::cerr << "Error: Matrix singular or near-singular at " << k << std::endl;
            return;
        }
        if (max_idx != k) {
            std::swap(pivot[k], pivot[max_idx]);
            for (int j = 0; j < n; ++j) {
                std::swap(A[k * n + j], A[max_idx * n + j]);
            }
        }

        for (int i = k + 1; i < n; ++i) {
            A[i * n + k] /= A[k * n + k];
            for (int j = k + 1; j < n; ++j) {
                A[i * n + j] -= A[i * n + k] * A[k * n + j];
            }
        }
    }
}

double* forward_substitution(const double* LU, const double* b, const int* pivot, int n) {
    double* y = new double[n]();
    for (int i = 0; i < n; ++i) {
        int pi = pivot[i];
        y[i] = b[pi];
        for (int j = 0; j < i; ++j) {
            y[i] -= LU[i * n + j] * y[j];
        }
    }
    return y;
}

double* backward_substitution(const double* LU, const double* y, int n) {
    double* x = new double[n]();
    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= LU[i * n + j] * x[j];
        }
        x[i] /= LU[i * n + i];

    }
    return x;
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

float dot(const double* a, const double* b, int n) {
    float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double* axpy(float alpha, const double* x, const double* y, int n) {
    double* result = new double[n];
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

float norm(const double* v, int n) {
    float d = dot(v, v, n);
    // if (isnan(d) || isinf(d)) return -1.0;
    return sqrt(d);
}

struct IRResult {
    double* x;
    float residual;
    int iterations;
};

IRResult iterative_refinement(const CSRMatrix& A, const double* b, double* LU, const int* pivot, 
                              int max_iter = 1000, float tol = 1e-12) {
    int n = A.n;
    double* x = new double[n]();
    double* r = new double[n];
    for (int i = 0; i < n; ++i) r[i] = b[i];
    
    float initial_norm = norm(b, n);
    if (initial_norm < 0) {
        //std::cerr << "Error: Initial b has invalid norm" << std::endl;
        IRResult result = {x, -1.0, 0};
        delete[] r;
        return result;
    }
    float tol_abs = tol * initial_norm;

    int k;
    for (k = 0; k < max_iter; ++k) {
        float r_norm = norm(r, n);
        if (r_norm < 0) {
            std::cerr << "Error: Residual became NaN or Inf at iteration " << k << std::endl;
            break;
        }
        if (r_norm < tol_abs) break;

        double* y = forward_substitution(LU, r, pivot, n);
        double* d = backward_substitution(LU, y, n);
        double* new_x = axpy(1.0, d, x, n);
        delete[] x;
        x = new_x;
        double* Ax = matvec(A, x);
        double* new_r = axpy(-1.0, Ax, b, n);
        delete[] r;
        delete[] y;
        delete[] d;
        delete[] Ax;
        r = new_r;
    }

    float residual = norm(r, n);
    delete[] r;
    return {x, residual, k};
}

double* generate_rhs(int n) {
    double* b = new double[n];
    std::mt19937 gen(0);
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}


int main() {
    std::string filename = "1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) return 1;

    double* LU = csr_to_dense(A);
    int* pivot = new int[A.n];
    lu_factorize_with_pivoting(LU, pivot, A.n);

    double* b = generate_rhs(A.n);

    IRResult result = iterative_refinement(A, b, LU, pivot, A.n);
    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    double* Ax = matvec(A, result.x);
    double* resid = axpy(-1.0, Ax, b, A.n);
    float verify_residual = norm(resid, A.n);
    std::cout << "Verification residual: " << verify_residual << std::endl;

    double solution[A.n];
    for (int i=0; i<A.n; i++){
        solution[i] = result.x[i];
    }

    PROMISE_CHECK_ARRAY(solution, A.n);

    delete[] LU;
    delete[] pivot;
    delete[] b;
    delete[] result.x;
    delete[] Ax;
    delete[] resid;
    free_csr_matrix(A);

    return 0;
}