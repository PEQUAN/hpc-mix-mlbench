#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>

struct CSRMatrix {
    int n;
    std::vector<__PROMISE__> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
};

bool compare_by_column(const std::pair<int, __PROMISE__>& a, const std::pair<int, __PROMISE__>& b) {
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

    std::vector<std::vector<std::pair<int, __PROMISE__>>> temp(n);
    for (int k = 0; k < nz; ++k) {
        if (!getline(file, line)) {
            std::cerr << "Error: Unexpected end of file" << std::endl;
            return A;
        }
        ss.clear();
        ss.str(line);
        int i, j;
        __PROMISE__ val;
        ss >> i >> j >> val;
        i--; j--;
        temp[i].push_back({j, val});
        if (i != j) temp[j].push_back({i, val});
    }

    A.row_ptr.resize(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        std::sort(temp[i].begin(), temp[i].end(), compare_by_column);
        A.row_ptr[i + 1] = A.row_ptr[i] + temp[i].size();
        for (const auto& entry : temp[i]) {
            A.col_indices.push_back(entry.first);
            A.values.push_back(entry.second);
        }
    }

    std::cout << "Loaded matrix: " << n << " x " << n << " with " << A.values.size() << " non-zeros" << std::endl;
    return A;
}

std::vector<std::vector<__PROMISE__>> csr_to_dense(const CSRMatrix& A) {
    std::vector<std::vector<__PROMISE__>> dense(A.n, std::vector<__PROMISE__>(A.n, 0.0));
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            dense[i][A.col_indices[j]] = A.values[j];
        }
    }
    return dense;
}

void lu_factorize_with_pivoting(std::vector<std::vector<__PROMISE__>>& A, std::vector<int>& pivot) {
    int n = A.size();
    pivot.resize(n);
    for (int i = 0; i < n; ++i) pivot[i] = i;

    for (int k = 0; k < n; ++k) {
        // Find pivot
        __PROMISE__ max_val = abs(A[k][k]);
        int max_idx = k;
        for (int i = k + 1; i < n; ++i) {
            if (abs(A[i][k]) > max_val) {
                max_val = abs(A[i][k]);
                max_idx = i;
            }
        }
        if (max_val < 1e-10) {
            std::cerr << "Error: Matrix singular or near-singular at " << k << std::endl;
            return;
        }
        if (max_idx != k) {
            std::swap(pivot[k], pivot[max_idx]);
            std::swap(A[k], A[max_idx]);
        }

        for (int i = k + 1; i < n; ++i) {
            A[i][k] /= A[k][k];
            for (int j = k + 1; j < n; ++j) {
                A[i][j] -= A[i][k] * A[k][j];
            }
        }
    }
}

std::vector<__PROMISE__> forward_substitution(const std::vector<std::vector<__PROMISE__>>& LU, 
                                        const std::vector<__PROMISE__>& b, const std::vector<int>& pivot) {
    int n = LU.size();
    std::vector<__PROMISE__> y(n, 0.0);
    for (int i = 0; i < n; ++i) {
        int pi = pivot[i];
        y[i] = b[pi];
        for (int j = 0; j < i; ++j) {
            y[i] -= LU[i][j] * y[j];
        }
    }
    return y;
}

std::vector<__PROMISE__> backward_substitution(const std::vector<std::vector<__PROMISE__>>& LU, const std::vector<__PROMISE__>& y) {
    int n = LU.size();
    std::vector<__PROMISE__> x(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = y[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= LU[i][j] * x[j];
        }
        x[i] /= LU[i][i];
        // if (std::isnan(x[i]) || std::isinf(x[i])) {
        //    std::cerr << "Error: NaN/Inf in backward substitution at " << i << std::endl;
        //    return x;
        //}
    }
    return x;
}

std::vector<__PROMISE__> matvec(const CSRMatrix& A, const std::vector<__PROMISE__>& x) {
    std::vector<__PROMISE__> y(A.n, 0.0);
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return y;
}

__PROMISE__ dot(const std::vector<__PROMISE__>& a, const std::vector<__PROMISE__>& b) {
    __PROMISE__ sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

std::vector<__PROMISE__> axpy(__PROMISE__ alpha, const std::vector<__PROMISE__>& x, const std::vector<__PROMISE__>& y) {
    std::vector<__PROMISE__> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}

__PROMISE__ norm(const std::vector<__PROMISE__>& v) {
    __PROMISE__ d = dot(v, v);
    // if (std::isnan(d) || std::isinf(d)) return -1.0;
    return sqrt(d);
}

struct IRResult {
    std::vector<__PROMISE__> x;
    __PROMISE__ residual;
    int iterations;
};

IRResult iterative_refinement(const CSRMatrix& A, const std::vector<__PROMISE__>& b, 
                              std::vector<std::vector<__PROMISE__>>& LU, const std::vector<int>& pivot, 
                              int max_iter = 1000, __PROMISE__ tol = 1e-6) {
    int n = A.n;
    std::vector<__PROMISE__> x(n, 0.0);
    std::vector<__PROMISE__> r = b;
    __PROMISE__ initial_norm = norm(b);
    if (initial_norm < 0) {
        std::cerr << "Error: Initial b has invalid norm" << std::endl;
        return {x, -1.0, 0};
    }
    __PROMISE__ tol_abs = tol * initial_norm;

    int k;
    for (k = 0; k < max_iter; ++k) {
        __PROMISE__ r_norm = norm(r);
        if (r_norm < 0) {
            std::cerr << "Error: Residual became NaN or Inf at iteration " << k << std::endl;
            break;
        }
        if (r_norm < tol_abs) break;

        std::vector<__PROMISE__> y = forward_substitution(LU, r, pivot);
        std::vector<__PROMISE__> d = backward_substitution(LU, y);
        x = axpy(1.0, d, x);
        r = axpy(-1.0, matvec(A, x), b);
    }

    __PROMISE__ residual = norm(r);
    return {x, residual, k};
}

std::vector<__PR_1__> generate_rhs(int n) {
    std::vector<__PR_1__> b(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

void write_solution(const std::vector<__PROMISE__>& x, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file: " << filename << ". Check permissions or path." << std::endl;
        return;
    }
    file << "x\n";
    for (__PROMISE__ val : x) {
        file << val << "\n";
    }
    file.close();
}

int main() {
    std::string filename = "1138_bus.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) return 1;

    std::vector<std::vector<__PROMISE__>> LU = csr_to_dense(A);
    std::vector<int> pivot;
    lu_factorize_with_pivoting(LU, pivot);

    std::vector<__PROMISE__> b = generate_rhs(A.n);

    auto start = std::chrono::high_resolution_clock::now();
    IRResult result = iterative_refinement(A, b, LU, pivot, A.n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;
    PROMISE_CHECK_VAR(result.residual);
    std::vector<__PROMISE__> Ax = matvec(A, result.x);
    __PROMISE__ verify_residual = norm(axpy(-1.0, Ax, b));
    std::cout << "Verification residual: " << verify_residual << std::endl;
    // PROMISE_CHECK_ARRAY(result.x.data(), A.n);
    // write_solution(result.x, "results/iterative_refinement/ir_solution.csv");

    return 0;
}
