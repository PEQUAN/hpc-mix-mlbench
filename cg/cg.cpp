#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <omp.h>

struct CSRMatrix {
    int n;
    double* values;
    int* col_indices;
    int* row_ptr;
    int nnz; // number of non-zeros
};

bool compare_by_column(const std::pair<int, double>& a, const std::pair<int, double>& b) {
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
        std::cerr << "Error: Matrix must be square" << std::endl;
        return A;
    }
    A.n = n;

    struct Entry { int row, col; double val; };
    std::vector<Entry> entries(2 * nz);
    int entry_count = 0;

    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        double val;
        ss >> i >> j >> val;
        if (i < 1 || j < 1 || i > n || j > n) {
            std::cerr << "Error: Invalid indices in Matrix Market file" << std::endl;
            return A;
        }
        i--; j--;
        entries[entry_count++] = {i, j, val};
        if (i != j) entries[entry_count++] = {j, i, val};
    }
    entries.resize(entry_count);

    std::vector<int> nnz_per_row(n, 0);
    for (int k = 0; k < entry_count; ++k) {
        nnz_per_row[entries[k].row]++;
    }

    A.nnz = entry_count;
    A.values = new double[entry_count];
    A.col_indices = new int[entry_count];
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row[i];
    }

    std::sort(entries.begin(), entries.end(),
        [](const Entry& a, const Entry& b) {
            return a.row == b.row ? a.col < b.col : a.row < b.row;
        });

    for (int k = 0; k < entry_count; ++k) {
        A.col_indices[k] = entries[k].col;
        A.values[k] = entries[k].val;
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << entry_count << " non-zeros" << std::endl;

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

void matvec(const CSRMatrix& A, const double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < A.n; ++i) {
        y[i] = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
}

double dot(const double* a, const double* b, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void axpy(double alpha, const double* x, const double* y, int n, double* result) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
}

double norm(const double* v, int n) {
    double d = dot(v, v, n);
    if (std::isnan(d) || std::isinf(d)) return -1.0;
    return std::sqrt(d);
}

void compute_diagonal_preconditioner(const CSRMatrix& A, double* M) {
    bool has_zero_diagonal = false;
    double min_diag = std::numeric_limits<double>::max();
    double max_diag = 0.0;
    for (int i = 0; i < A.n; ++i) {
        M[i] = 0.0;
        bool found_diag = false;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] == i) {
                M[i] = A.values[j];
                found_diag = true;
                min_diag = std::min(min_diag, std::abs(M[i]));
                max_diag = std::max(max_diag, std::abs(M[i]));
                break;
            }
        }
        if (!found_diag || std::abs(M[i]) < 1e-10) {
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

void apply_preconditioner(const double* M, const double* r, int n, double* z) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        z[i] = M[i] * r[i];
    }
}

struct Result {
    double* x;
    double residual;
    int iterations;
    std::vector<double> residual_history;
};

Result pcg(const CSRMatrix& A, const double* b, int max_iter = 1000, double tol = 1e-12) {
    int n = A.n;
    std::vector<double> x(n, 0.0);
    std::vector<double> r(n);
    for (int i = 0; i < n; ++i) r[i] = b[i]; // r = b - Ax (x = 0)
    std::vector<double> z(n);
    std::vector<double> M(n);
    compute_diagonal_preconditioner(A, M.data());
    apply_preconditioner(M.data(), r.data(), n, z.data()); // z = M^{-1} r
    std::vector<double> p(n);
    for (int i = 0; i < n; ++i) p[i] = z[i]; // p_0 = M^{-1} r_0
    std::vector<double> residual_history;

    double r_z = dot(r.data(), z.data(), n);
    double initial_norm = norm(b, n);
    if (initial_norm < 0) {
        std::cerr << "Error: Initial b has invalid norm" << std::endl;
        double* x_out = new double[n];
        for (int i = 0; i < n; ++i) x_out[i] = x[i];
        return {x_out, -1.0, 0, {}};
    }
    std::cout << "Initial norm of b: " << initial_norm << std::endl;
    double tol_abs = tol * initial_norm;

    int k;
    for (k = 0; k < max_iter; ++k) {
        std::vector<double> Ap(n);
        matvec(A, p.data(), Ap.data());
        double p_Ap = dot(p.data(), Ap.data(), n);
        if (std::abs(p_Ap) < 1e-10) {
            std::cerr << "Breakdown: p^T Ap = " << p_Ap << " at iteration " << k << std::endl;
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            double* r_temp = new double[n];
            matvec(A, x.data(), r_temp);
            axpy(-1.0, r_temp, b, n, r_temp);
            double* z_temp = new double[n];
            apply_preconditioner(M.data(), r_temp, n, z_temp);
            double r_norm = norm(z_temp, n);
            delete[] r_temp; delete[] z_temp;
            return {x_out, r_norm, k, residual_history};
        }
        double alpha = r_z / p_Ap;
        axpy(alpha, p.data(), x.data(), n, x.data());
        axpy(-alpha, Ap.data(), r.data(), n, r.data());
        apply_preconditioner(M.data(), r.data(), n, z.data());
        double r_norm = norm(z.data(), n);
        residual_history.push_back(r_norm);
        if (r_norm < tol_abs) {
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            return {x_out, r_norm, k + 1, residual_history};
        }
        double r_z_new = dot(r.data(), z.data(), n);
        if (std::abs(r_z_new) < 1e-10) {
            std::cerr << "Breakdown: r^T z = " << r_z_new << " at iteration " << k << std::endl;
            double* x_out = new double[n];
            for (int i = 0; i < n; ++i) x_out[i] = x[i];
            double* r_temp = new double[n];
            matvec(A, x.data(), r_temp);
            axpy(-1.0, r_temp, b, n, r_temp);
            double* z_temp = new double[n];
            apply_preconditioner(M.data(), r_temp, n, z_temp);
            double r_norm = norm(z_temp, n);
            delete[] r_temp; delete[] z_temp;
            return {x_out, r_norm, k, residual_history};
        }
        double beta = r_z_new / r_z;
        axpy(beta, p.data(), z.data(), n, p.data());
        r_z = r_z_new;
        if (k % 100 == 0) {
            std::cout << "Iteration " << k << ": Preconditioned Residual = " << r_norm << std::endl;
        }
    }

    double* r_temp = new double[n];
    matvec(A, x.data(), r_temp);
    axpy(-1.0, r_temp, b, n, r_temp);
    double* z_temp = new double[n];
    apply_preconditioner(M.data(), r_temp, n, z_temp);
    double r_norm = norm(z_temp, n);
    delete[] r_temp; delete[] z_temp;
    double* x_out = new double[n];
    for (int i = 0; i < n; ++i) x_out[i] = x[i];
    return {x_out, r_norm, k + 1, residual_history};
}

double* generate_rhs(const CSRMatrix& A) {
    std::vector<double> x_true(A.n, 1.0); // x_true = [1, 1, ..., 1]
    double* b = new double[A.n];
    matvec(A, x_true.data(), b);
    std::cout << "Generated b = A * x_true, where x_true = [1, 1, ..., 1]" << std::endl;
    return b;
}

void write_solution(const double* x, int n, const std::string& filename, const std::vector<double>& residual_history) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << ". Check permissions or path." << std::endl;
        return;
    }
    file << "x\n";
    for (int i = 0; i < n; ++i) {
        file << x[i] << "\n";
    }
    file << "\nResidual History\n";
    for (size_t i = 0; i < residual_history.size(); ++i) {
        file << i << "," << residual_history[i] << "\n";
    }
    file.close();
}

int main() {
    std::string filename = "../data/suitesparse/1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) {
        free_csr_matrix(A);
        return 1;
    }

    double* b = generate_rhs(A);

    auto start = std::chrono::high_resolution_clock::now();
    Result result = pcg(A, b, 2 * A.n, 1e-8);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final preconditioned residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

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

    write_solution(result.x, A.n, "pcg_solution.csv", result.residual_history);

    delete[] b;
    delete[] result.x;
    free_csr_matrix(A);
    return 0;
}