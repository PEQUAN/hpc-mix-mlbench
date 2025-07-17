#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <stdexcept>

struct CSRMatrix {
    int n;
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
    int nnz;
};

struct Entry { int row, col; double val; };

struct Result {
    std::vector<double> x;
    double residual;
    int iterations;
    std::vector<double> residual_history;
};

CSRMatrix read_mtx_file(const std::string& filename) {
    CSRMatrix A = {0, {}, {}, {}, 0};
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open " + filename);
    }

    std::string line;
    while (std::getline(file, line) && line[0] == '%') {}

    std::stringstream ss(line);
    int n, m, nz;
    ss >> n >> m >> nz;
    if (n != m) {
        throw std::runtime_error("Matrix must be square");
    }
    A.n = n;

    std::vector<Entry> entries;
    entries.reserve(2 * nz);
    std::vector<int> nnz_per_row(n, 0);

    for (int k = 0; k < nz; ++k) {
        if (!std::getline(file, line)) {
            throw std::runtime_error("Unexpected end of file");
        }
        ss.clear();
        ss.str(line);
        int i, j;
        double val;
        ss >> i >> j >> val;
        if (i < 1 || j < 1 || i > n || j > n) {
            throw std::runtime_error("Invalid indices in Matrix Market file");
        }
        i--; j--;
        entries.push_back({i, j, val});
        if (i != j) entries.push_back({j, i, val});
        nnz_per_row[i]++;
        if (i != j) nnz_per_row[j]++;
    }

    A.nnz = entries.size();
    A.values.resize(A.nnz);
    A.col_indices.resize(A.nnz);
    A.row_ptr.resize(n + 1);
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; ++i) {
        A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row[i];
    }

    std::sort(entries.begin(), entries.end(),
        [](const Entry& a, const Entry& b) {
            return a.row == b.row ? a.col < b.col : a.row < b.row;
        });

    for (int k = 0; k < A.nnz; ++k) {
        A.col_indices[k] = entries[k].col;
        A.values[k] = entries[k].val;
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << A.nnz << " non-zeros" << std::endl;
    return A;
}

void matvec(const CSRMatrix& A, const double* x, double* y) {
    #pragma omp parallel for schedule(dynamic) if (A.n > 1000)
    for (int i = 0; i < A.n; ++i) {
        y[i] = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
}

std::vector<double> generate_rhs(const CSRMatrix& A) {
    std::vector<double> x_true(A.n, 1.0); // x_true = [1, 1, ..., 1]
    std::vector<double> b(A.n, 0.0);
    matvec(A, x_true.data(), b.data());
    std::cout << "Generated b = A * x_true, where x_true = [1, 1, ..., 1]" << std::endl;
    return b;
}

double dot(const double* a, const double* b, int n) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) if (n > 1000)
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void axpy(double alpha, const double* x, const double* y, int n, double* result) {
    #pragma omp parallel for if (n > 1000)
    for (int i = 0; i < n; ++i) {
        result[i] = alpha * x[i] + y[i];
    }
}

double norm(const double* v, int n) {
    double d = dot(v, v, n);
    if (std::isnan(d) || std::isinf(d) || d < 0.0) {
        throw std::runtime_error("Invalid norm");
    }
    return std::sqrt(d);
}

void compute_diagonal_preconditioner(const CSRMatrix& A, std::vector<double>& M) {
    M.resize(A.n, 1.0);
    double min_diag = std::numeric_limits<double>::max();
    double max_diag = 0.0;
    bool has_zero_diagonal = false;

    for (int i = 0; i < A.n; ++i) {
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
    #pragma omp parallel for if (n > 1000)
    for (int i = 0; i < n; ++i) {
        z[i] = M[i] * r[i];
    }
}

void write_solution(const std::vector<double>& x, const std::string& filename,
                    const std::vector<double>& residual_history) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open output file: " + filename);
    }
    file << "Index,Solution\n";
    for (size_t i = 0; i < x.size(); ++i) {
        file << i << "," << x[i] << "\n";
    }
    file << "\nIteration,Residual\n";
    for (size_t i = 0; i < residual_history.size(); ++i) {
        file << i << "," << residual_history[i] << "\n";
    }
    file.close();
}

void arnoldi_step(const CSRMatrix& A, const double* M, double* V, double* H, int j,
                  int n, double* z, double* w, double initial_norm, int restart) {
    apply_preconditioner(M, &V[j * n], n, z);
    matvec(A, z, w);
    for (int i = 0; i <= j; ++i) {
        double h_ij = dot(w, &V[i * n], n);
        H[i * restart + j] = h_ij;
        axpy(-h_ij, &V[i * n], w, n, w);
    }
    double h_jp1_j = norm(w, n);
    H[(j + 1) * restart + j] = h_jp1_j;
    if (h_jp1_j < 1e-12 * initial_norm) {
        throw std::runtime_error("Arnoldi breakdown at iteration " + std::to_string(j));
    }
    #pragma omp parallel for if (n > 1000)
    for (int i = 0; i < n; ++i) {
        V[(j + 1) * n + i] = w[i] / h_jp1_j;
    }
}

Result gmres(const CSRMatrix& A, const std::vector<double>& b, int max_iter, double tol, int restart) {
    int n = A.n;
    std::vector<double> x(n, 0.0);
    std::vector<double> r(n);
    std::vector<double> M(n);
    std::vector<double> residual_history;

    compute_diagonal_preconditioner(A, M);

    // Initial residual: r = b
    std::copy(b.begin(), b.end(), r.begin());
    double initial_norm = norm(r.data(), n);
    std::cout << "Initial norm of residual: " << initial_norm << std::endl;
    double tol_abs = tol * std::max(initial_norm, 1e-10);

    std::vector<double> V(n * (restart + 1));
    std::vector<double> H((restart + 1) * restart, 0.0);
    std::vector<double> z(n);
    std::vector<double> w(n);
    int total_iterations = 0;

    while (total_iterations < max_iter) {
        // Compute true residual
        matvec(A, x.data(), r.data());
        axpy(-1.0, r.data(), b.data(), n, r.data());
        double r_norm = norm(r.data(), n);
        residual_history.push_back(r_norm);
        if (r_norm < tol_abs || r_norm < 0 || std::isnan(r_norm) || std::isinf(r_norm)) {
            break;
        }

        // Initialize V[:,0]
        for (int i = 0; i < n; ++i) V[i] = r[i] / r_norm;
        std::vector<double> g(restart + 1, 0.0);
        g[0] = r_norm;
        std::vector<double> cs(restart), sn(restart);

        int j;
        try {
            for (j = 0; j < restart && total_iterations < max_iter; ++j) {
                arnoldi_step(A, M.data(), V.data(), H.data(), j, n, z.data(), w.data(), initial_norm, restart);

                // Apply Givens rotations
                for (int i = 0; i < j; ++i) {
                    double temp = cs[i] * H[i * restart + j] + sn[i] * H[(i + 1) * restart + j];
                    H[(i + 1) * restart + j] = -sn[i] * H[i * restart + j] + cs[i] * H[(i + 1) * restart + j];
                    H[i * restart + j] = temp;
                }
                double a = H[j * restart + j];
                double b1 = H[(j + 1) * restart + j];
                double rho = std::sqrt(a * a + b1 * b1);
                if (rho < 1e-12 * initial_norm) {
                    throw std::runtime_error("Givens rotation breakdown");
                }
                cs[j] = a / rho;
                sn[j] = b1 / rho;
                H[j * restart + j] = rho;
                H[(j + 1) * restart + j] = 0.0;

                double temp = cs[j] * g[j] + sn[j] * g[j + 1];
                g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1];
                g[j] = temp;

                r_norm = std::abs(g[j + 1]);
                residual_history.push_back(r_norm);
                total_iterations++;

                if (total_iterations % 100 == 0) {
                    std::cout << "Iteration " << total_iterations << ": Residual = " << r_norm << std::endl;
                }
                if (r_norm < tol_abs) {
                    j++;
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: " << e.what() << std::endl;
            // Continue with current solution
        }

        // Solve least-squares problem
        std::vector<double> y(j, 0.0);
        for (int i = j - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int k = i + 1; k < j; ++k) {
                y[i] -= H[i * restart + k] * y[k];
            }
            if (std::abs(H[i * restart + i]) < 1e-12 * initial_norm) {
                std::cerr << "Warning: Least-squares breakdown at iteration " << total_iterations << std::endl;
                break;
            }
            y[i] /= H[i * restart + i];
        }

        // Update solution
        for (int k = 0; k < j; ++k) {
            apply_preconditioner(M.data(), &V[k * n], n, z.data());
            axpy(y[k], z.data(), x.data(), n, x.data());
        }

        if (r_norm < tol_abs) {
            break;
        }
    }

    // Final residual
    matvec(A, x.data(), r.data());
    axpy(-1.0, r.data(), b.data(), n, r.data());
    double r_norm = norm(r.data(), n);
    residual_history.push_back(r_norm);

    return {x, r_norm, total_iterations, residual_history};
}

int main(int argc, char* argv[]) {
    try {
        std::string filename = (argc > 1) ? argv[1] : "../data/suitesparse/1138_bus.mtx";
        int max_iter = (argc > 2) ? std::stoi(argv[2]) : 1000;
        double tol = (argc > 3) ? std::stod(argv[3]) : 1e-12;
        int restart = (argc > 4) ? std::stoi(argv[4]) : 1000;

        CSRMatrix A = read_mtx_file(filename);
        std::vector<double> b = generate_rhs(A);

        auto start = std::chrono::high_resolution_clock::now();
        Result result = gmres(A, b, max_iter, tol, restart);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
        std::cout << "Training time: " << duration.count() << " ms" << std::endl;
        std::cout << "Final residual: " << result.residual << std::endl;
        std::cout << "Iterations to converge: " << result.iterations << std::endl;

        // Verify solution
        std::vector<double> r(A.n);
        matvec(A, result.x.data(), r.data());
        axpy(-1.0, r.data(), b.data(), A.n, r.data());
        double verify_residual = norm(r.data(), A.n);
        std::cout << "Verification residual: " << verify_residual << std::endl;

        // Print first few solution entries
        std::cout << "First 5 solution entries:\n";
        for (int i = 0; i < std::min(5, A.n); ++i) {
            std::cout << "x[" << i << "] = " << result.x[i] << std::endl;
        }

        write_solution(result.x, "gmres_solution.csv", result.residual_history);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}