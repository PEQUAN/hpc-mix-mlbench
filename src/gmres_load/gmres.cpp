#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>

struct CSRMatrix {
    int n;
    double* values;
    int* col_indices;
    int* row_ptr;
    int nnz; // Store number of non-zero elements
};

struct Pair {
    int first;
    double second;
};

bool compare_by_column(const Pair& a, const Pair& b) {
    return a.first < b.first;
}

void free_csr_matrix(CSRMatrix& A) {
    delete[] A.values;
    delete[] A.col_indices;
    delete[] A.row_ptr;
    A.values = nullptr;
    A.col_indices = nullptr;
    A.row_ptr = nullptr;
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

    // Temporary storage for entries
    Pair** temp = new Pair*[n]();
    int* temp_sizes = new int[n]();
    for (int i = 0; i < n; ++i) {
        temp[i] = new Pair[nz](); // Over-allocate to be safe
    }

    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            for (int i = 0; i < n; ++i) delete[] temp[i];
            delete[] temp;
            delete[] temp_sizes;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        double val;
        ss >> i >> j >> val;
        i--; j--;
        temp[i][temp_sizes[i]] = {j, val};
        temp_sizes[i]++;
        if (i != j) {
            temp[j][temp_sizes[j]] = {i, val};
            temp_sizes[j]++;
        }
    }

    // Count total non-zeros
    A.nnz = 0;
    for (int i = 0; i < n; ++i) {
        A.nnz += temp_sizes[i];
    }

    // Allocate CSR arrays
    A.values = new double[A.nnz];
    A.col_indices = new int[A.nnz];
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;

    // Fill CSR arrays
    int pos = 0;
    for (int i = 0; i < n; ++i) {
        std::sort(temp[i], temp[i] + temp_sizes[i], compare_by_column);
        A.row_ptr[i + 1] = A.row_ptr[i] + temp_sizes[i];
        for (int k = 0; k < temp_sizes[i]; ++k) {
            A.col_indices[pos] = temp[i][k].first;
            A.values[pos] = temp[i][k].second;
            pos++;
        }
    }

    // Clean up temporary arrays
    for (int i = 0; i < n; ++i) delete[] temp[i];
    delete[] temp;
    delete[] temp_sizes;

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << A.nnz << " non-zeros" << std::endl;
    return A;
}

double* matvec(const CSRMatrix& A, const double* x) {
    double* y = new double[A.n]();
    for (int i = 0; i < A.n; ++i) {
        double sum = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            sum += A.values[j] * x[A.col_indices[j]];
        }
        y[i] = sum;
    }
    return y;
}

double dot(const double* a, const double* b, int n) {
    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

double* axpy(double alpha, const double* x, const double* y, int n) {
    double* result = new double[n];
    for (int i = 0; i < n; ++i) {
        result[i] = std::fma(alpha, x[i], y[i]);
    }
    return result;
}

double norm(const double* v, int n) {
    double d = dot(v, v, n);
    return std::sqrt(d);
}

double* solve_hessenberg(double** H, double beta, int m, int n) {
    double* y = new double[m]();
    double* rhs = new double[m + 1]();
    rhs[0] = beta;
    double** R = new double*[m + 1];
    for (int i = 0; i < m + 1; ++i) {
        R[i] = new double[m]();
        for (int j = 0; j < m; ++j) {
            R[i][j] = H[i][j];
        }
    }
    
    const double eps = std::numeric_limits<double>::epsilon();
    
    for (int i = 0; i < m; ++i) {
        if (i + 1 >= m + 1) break;
        double hii = R[i][i];
        double hi1i = R[i + 1][i];
        double r = std::hypot(hii, hi1i);
        
        if (r < eps) continue;
        
        double c = hii / r;
        double s = -hi1i / r;
        
        for (int j = i; j < m; ++j) {
            double t1 = R[i][j];
            double t2 = R[i + 1][j];
            R[i][j] = std::fma(c, t1, -s * t2);
            R[i + 1][j] = std::fma(s, t1, c * t2);
        }
        
        double t1 = rhs[i];
        double t2 = rhs[i + 1];
        rhs[i] = std::fma(c, t1, -s * t2);
        rhs[i + 1] = std::fma(s, t1, c * t2);
    }
    
    for (int i = m - 1; i >= 0; --i) {
        if (std::abs(R[i][i]) < eps) {
            y[i] = 0.0;
            continue;
        }
        y[i] = rhs[i];
        for (int j = i + 1; j < m; ++j) {
            y[i] -= R[i][j] * y[j];
        }
        y[i] /= R[i][i];
    }
    
    for (int i = 0; i < m + 1; ++i) delete[] R[i];
    delete[] R;
    delete[] rhs;
    return y;
}

struct GMRESResult {
    double* x;
    double residual;
    int iterations;
    bool converged;
};

GMRESResult gmres(const CSRMatrix& A, const double* b, int max_iter = 1000, double tol = 1e-6) {
    const double eps = std::numeric_limits<double>::epsilon();
    int n = A.n;
    double* x = new double[n]();
    double* r = new double[n];
    for (int i = 0; i < n; ++i) r[i] = b[i];
    double beta = norm(r, n);
    
    if (beta < eps) {
        GMRESResult result = {x, beta, 0, true};
        delete[] r;
        return result;
    }
    
    double tol_abs = tol * beta;
    double** V = new double*[max_iter + 1];
    double** H = new double*[max_iter + 1];
    for (int i = 0; i < max_iter + 1; ++i) {
        V[i] = nullptr;
        H[i] = new double[max_iter]();
    }
    
    V[0] = axpy(1.0 / beta, r, new double[n](), n);
    int k;
    
    for (k = 0; k < max_iter; ++k) {
        double* w = matvec(A, V[k]);
        for (int i = 0; i <= k; ++i) {
            double h_ik = dot(V[i], w, n);
            H[i][k] = h_ik;
            double* temp = axpy(-h_ik, V[i], w, n);
            delete[] w;
            w = temp;
        }
        
        double h_next = norm(w, n);
        if (k + 1 < max_iter + 1) H[k + 1][k] = h_next;
        
        if (h_next < eps * beta) {
            std::cout << "Breakdown at iteration " << k << std::endl;
            break;
        }
        
        V[k + 1] = axpy(1.0 / h_next, w, new double[n](), n);
        delete[] w;
        
        double* y = solve_hessenberg(H, beta, k + 1, max_iter);
        delete[] x;
        x = new double[n]();
        for (int j = 0; j <= k; ++j) {
            double* temp = axpy(y[j], V[j], x, n);
            delete[] x;
            x = temp;
        }
        
        delete[] r;
        r = axpy(-1.0, matvec(A, x), b, n);
        double r_norm = norm(r, n);
        
        delete[] y;
        if (r_norm < tol_abs) {
            std::cout << "Converged at iteration " << k + 1 << std::endl;
            for (int i = 0; i < max_iter + 1; ++i) delete[] H[i];
            delete[] H;
            for (int i = 0; i <= k + 1; ++i) delete[] V[i];
            delete[] V;
            delete[] r;
            return {x, r_norm, k + 1, true};
        }
    }
    
    std::cout << "Max iterations reached: " << k << std::endl;
    for (int i = 0; i < max_iter + 1; ++i) delete[] H[i];
    delete[] H;
    for (int i = 0; i <= k + 1; ++i) delete[] V[i];
    delete[] V;
    delete[] r;
    return {x, norm(x, n), k + 1, false};
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
        std::cerr << "Error opening output file: " << filename << std::endl;
        return;
    }
    file << "x\n";
    for (int i = 0; i < n; ++i) {
        file << x[i] << "\n";
    }
    file.close();
}

int main() {
    try {
        std::string filename = "../data/suitesparse/1138_bus.mtx";
        CSRMatrix A = read_mtx_file(filename);
        if (A.n == 0) {
            free_csr_matrix(A);
            return 1;
        }

        double* b = generate_rhs(A.n);

        auto start = std::chrono::high_resolution_clock::now();
        GMRESResult result = gmres(A, b, A.n);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
        std::cout << "Time: " << duration.count() << " ms" << std::endl;
        std::cout << "Final residual: " << result.residual << std::endl;
        std::cout << "Iterations: " << result.iterations << std::endl;
        std::cout << "Converged: " << (result.converged ? "yes" : "no") << std::endl;

        write_solution(result.x, A.n, "../results/gmres2/gmres_solution.csv");

        // Clean up
        free_csr_matrix(A);
        delete[] b;
        delete[] result.x;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}