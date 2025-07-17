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

// ILU(0) factorization: L and U stored in a single CSR matrix, with L having 1s on diagonal
CSRMatrix compute_ilu_factorization(const CSRMatrix& A) {
    CSRMatrix LU = A; // Copy structure
    LU.values.assign(A.values.begin(), A.values.end()); // Copy values

    for (int i = 0; i < A.n; ++i) {
        // Find diagonal element
        double diag = 0.0;
        int diag_idx = -1;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] == i) {
                diag = A.values[j];
                diag_idx = j;
                break;
            }
        }
        if (std::abs(diag) < 1e-10) {
            throw std::runtime_error("Zero or near-zero diagonal at row " + std::to_string(i));
        }

        // Update LU values for row i
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            if (A.col_indices[j] < i) {
                // Lower triangular part (L)
                LU.values[j] /= diag;
            } else if (A.col_indices[j] == i) {
                // Diagonal (U)
                LU.values[j] = diag;
            }
        }

        // Update subsequent rows
        for (int k = i + 1; k < A.n; ++k) {
            // Find L_ki (if k,i exists in sparsity pattern)
            double lik = 0.0;
            int lik_idx = -1;
            for (int j = A.row_ptr[k]; j < A.row_ptr[k + 1]; ++j) {
                if (A.col_indices[j] == i) {
                    lik = A.values[j] / diag;
                    lik_idx = j;
                    break;
                }
            }
            if (lik_idx == -1) continue; // Skip if no entry at (k,i)

            // Update row k in U
            for (int j = A.row_ptr[k]; j < A.row_ptr[k + 1]; ++j) {
                if (A.col_indices[j] <= i) continue; // Skip L part and diagonal
                for (int m = A.row_ptr[i]; m < A.row_ptr[i + 1]; ++m) {
                    if (A.col_indices[m] == A.col_indices[j]) {
                        LU.values[j] -= lik * LU.values[m];
                        break;
                    }
                }
            }
            LU.values[lik_idx] = lik; // Store L_ki
        }
    }

    return LU;
}

// Forward solve: Lz = r (L has 1s on diagonal)
void forward_solve(const CSRMatrix& LU, const double* r, double* z, int n) {
    std::vector<double> temp(n, 0.0);
    for (int i = 0; i < n; ++i) {
        temp[i] = r[i];
        for (int j = LU.row_ptr[i]; j < LU.row_ptr[i + 1]; ++j) {
            if (LU.col_indices[j] < i) {
                temp[i] -= LU.values[j] * temp[LU.col_indices[j]];
            }
        }
        // Diagonal of L is 1, so z[i] = temp[i]
        z[i] = temp[i];
    }
}

// Backward solve: Ux = z
void backward_solve(const CSRMatrix& LU, const double* z, double* x, int n) {
    std::vector<double> temp(z, z + n);
    for (int i = n - 1; i >= 0; --i) {
        double diag = 0.0;
        int diag_idx = -1;
        for (int j = LU.row_ptr[i]; j < LU.row_ptr[i + 1]; ++j) {
            if (LU.col_indices[j] == i) {
                diag = LU.values[j];
                diag_idx = j;
                break;
            }
        }
        if (std::abs(diag) < 1e-10) {
            throw std::runtime_error("Zero diagonal in U at row " + std::to_string(i));
        }
        x[i] = temp[i] / diag;
        for (int j = LU.row_ptr[i]; j < LU.row_ptr[i + 1]; ++j) {
            if (LU.col_indices[j] > i) {
                for (int k = LU.row_ptr[LU.col_indices[j]]; k < LU.row_ptr[LU.col_indices[j] + 1]; ++k) {
                    if (LU.col_indices[k] >= LU.col_indices[j]) {
                        temp[LU.col_indices[j]] -= LU.values[j] * x[i];
                        break;
                    }
                }
            }
        }
    }
}

// Apply ILU preconditioner: solve LUz = r
void apply_ilu_preconditioner(const CSRMatrix& LU, const double* r, double* z, int n) {
    std::vector<double> temp(n);
    forward_solve(LU, r, temp.data(), n);
    backward_solve(LU, temp.data(), z, n);
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

void arnoldi_step(const CSRMatrix& A, const CSRMatrix& LU, double* V, double* H, int j, int n,
                  double* w, double* z, double initial_norm, int restart) {
    // Apply preconditioner: solve LUz = V[:,j]
    apply_ilu_preconditioner(LU, &V[j * n], z, n);
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
    std::vector<double> residual_history;

    // Compute ILU factorization
    CSRMatrix LU = compute_ilu_factorization(A);

    // Initial residual: r = b
    std::copy(b.begin(), b.end(), r.begin());
    double initial_norm = norm(r.data(), n);
    std::cout << "Initial norm of residual: " << initial_norm << std::endl;
    double tol_abs = tol * std::max(initial_norm, 1e-10);

    std::vector<double> V(n * (restart + 1));
    std::vector<double> H((restart + 1) * restart, 0.0);
    std::vector<double> w(n);
    std::vector<double> z(n);
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
                arnoldi_step(A, LU, V.data(), H.data(), j, n, w.data(), z.data(), initial_norm, restart);

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

        // Update solution: x = x + M^-1 * V[:,0:j] * y
        for (int k = 0; k < j; ++k) {
            apply_ilu_preconditioner(LU, &V[k * n], z.data(), n);
            axpy(y[k], z.data(), x.data(), n, x.data());
        }

        if (r_norm < tol_abs) {
            break;
        }
    }

    
    matvec(A, x.data(), r.data()); // Final residual
    axpy(-1.0, r.data(), b.data(), n, r.data());
    double r_norm = norm(r.data(), n);
    residual_history.push_back(r_norm);

    return {x, r_norm, total_iterations, residual_history};
}

int main(int argc, char* argv[]) {
    try {
        std::string filename = (argc > 1) ? argv[1] : "../data/suitesparse/1138_bus.mtx";
        int max_iter = (argc > 2) ? std::stoi(argv[2]) : 1000;
        double tol = (argc > 3) ? std::stod(argv[3]) : 1e-24;
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