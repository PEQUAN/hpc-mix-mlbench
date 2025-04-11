#ifndef _Alignof
#define _Alignof(type) alignof(type)
#endif

#include <iostream>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <utility> // For std::pair

struct CSRMatrix {
    int n;          // Matrix dimension
    int nnz;        // Number of non-zeros
    double* values; // Non-zero values
    int* col_indices; // Column indices
    int* row_ptr;   // Row pointers (size n+1)

    CSRMatrix() : n(0), nnz(0), values(nullptr), col_indices(nullptr), row_ptr(nullptr) {}

    ~CSRMatrix() {
        delete[] values;
        delete[] col_indices;
        delete[] row_ptr;
    }
};

bool compare_by_column(const std::pair<int, double>& a, const std::pair<int, double>& b) {
    return a.first < b.first;
}

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

    // Temporary storage for entries
    struct RowEntry {
        std::pair<int, double>* entries;
        int size;
        int capacity;
    };
    RowEntry* temp = new RowEntry[n];
    for (int i = 0; i < n; ++i) {
        temp[i].size = 0;
        temp[i].capacity = 10; // Initial capacity
        temp[i].entries = new std::pair<int, double>[temp[i].capacity];
    }

    // Read non-zeros
    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            for (int i = 0; i < n; ++i) delete[] temp[i].entries;
            delete[] temp;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        double val;
        ss >> i >> j >> val;
        i--; j--; // Convert to 0-based indexing
        // Add to row i
        if (temp[i].size == temp[i].capacity) {
            temp[i].capacity *= 2;
            std::pair<int, double>* new_entries = new std::pair<int, double>[temp[i].capacity];
            for (int p = 0; p < temp[i].size; ++p) {
                new_entries[p] = temp[i].entries[p];
            }
            delete[] temp[i].entries;
            temp[i].entries = new_entries;
        }
        temp[i].entries[temp[i].size++] = {j, val};
        // Add to row j for symmetry (if off-diagonal)
        if (i != j) {
            if (temp[j].size == temp[j].capacity) {
                temp[j].capacity *= 2;
                std::pair<int, double>* new_entries = new std::pair<int, double>[temp[j].capacity];
                for (int p = 0; p < temp[j].size; ++p) {
                    new_entries[p] = temp[j].entries[p];
                }
                delete[] temp[j].entries;
                temp[j].entries = new_entries;
            }
            temp[j].entries[temp[j].size++] = {i, val};
        }
    }

    // Compute row_ptr and nnz
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    A.nnz = 0;
    for (int i = 0; i < n; ++i) {
        A.nnz += temp[i].size;
        A.row_ptr[i + 1] = A.row_ptr[i] + temp[i].size;
    }

    // Allocate values and col_indices
    A.values = new double[A.nnz];
    A.col_indices = new int[A.nnz];
    int pos = 0;
    for (int i = 0; i < n; ++i) {
        std::sort(temp[i].entries, temp[i].entries + temp[i].size, compare_by_column);
        for (int p = 0; p < temp[i].size; ++p) {
            A.col_indices[pos] = temp[i].entries[p].first;
            A.values[pos] = temp[i].entries[p].second;
            pos++;
        }
    }

    // Clean up temp
    for (int i = 0; i < n; ++i) {
        delete[] temp[i].entries;
    }
    delete[] temp;

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << A.nnz << " non-zeros" << std::endl;
    return A;
}

double* matvec(const CSRMatrix& A, const double* x, int n) {
    double* y = new double[n]();
    for (int i = 0; i < n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return y;
}

double* axpy(double alpha, const double* x, const double* y, int n) {
    double* result = new double[n];
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

double dot(const double* a, const double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

double norm(const double* v, int n) {
    return std::sqrt(dot(v, v, n));
}

struct CGResult {
    double* x;
    int n; // Size of x
    double residual;
    int iterations;

    ~CGResult() {
        delete[] x;
    }
};

CGResult conjugate_gradient(const CSRMatrix& A, const double* b, int max_iter = 1000, double tol = 1e-6) {
    int n = A.n;
    CGResult result;
    result.n = n;
    result.x = new double[n]();
    double* r = new double[n];
    for (int i = 0; i < n; ++i) r[i] = b[i];
    double* p = new double[n];
    for (int i = 0; i < n; ++i) p[i] = r[i];
    double rtr = dot(r, r, n);
    double tol2 = tol * tol * dot(b, b, n);

    int k;
    for (k = 0; k < max_iter && rtr > tol2; ++k) {
        double* Ap = matvec(A, p, n);
        double alpha = rtr / dot(p, Ap, n);
        double* x_new = axpy(alpha, p, result.x, n);
        delete[] result.x;
        result.x = x_new;
        double* r_new = axpy(-alpha, Ap, r, n);
        delete[] r;
        r = r_new;
        double rtr_new = dot(r, r, n);
        double beta = rtr_new / rtr;
        double* p_new = axpy(beta, p, r, n);
        delete[] p;
        p = p_new;
        delete[] Ap;
        rtr = rtr_new;
    }

    result.residual = norm(r, n);
    result.iterations = k;

    delete[] r;
    delete[] p;

    return result;
}

double* generate_rhs(int n) {
    double* b = new double[n];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

void write_solution(const double* x, int n, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file" << std::endl;
        return;
    }
    file << "x\n";
    for (int i = 0; i < n; ++i) {
        file << x[i] << "\n";
    }
}

int main() {
    std::string filename = "../data/suitesparse/rdb5000.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) return 1;

    double* b = generate_rhs(A.n);

    auto start = std::chrono::high_resolution_clock::now();
    CGResult result = conjugate_gradient(A, b, A.n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    double* Ax = matvec(A, result.x, A.n);
    double* verify_vec = axpy(-1.0, Ax, b, A.n);
    double verify_residual = norm(verify_vec, A.n);
    std::cout << "Verification residual: " << verify_residual << std::endl;

    write_solution(result.x, A.n, "../results/cg/cg_solution.csv");

    delete[] b;
    delete[] Ax;
    delete[] verify_vec;

    return 0;
}