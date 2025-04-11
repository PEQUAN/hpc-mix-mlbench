#ifndef _Alignof
#define _Alignof(type) alignof(type)
#endif

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <random>
#include <algorithm>

struct CSRMatrix {
    int n;
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
};

// Generate a guaranteed SPD matrix
CSRMatrix generate_random_symmetric_matrix(int n, double sparsity = 0.01) {
    CSRMatrix A;
    A.n = n;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> val_dis(0.1, 1.0);
    std::uniform_real_distribution<> prob_dis(0.0, 1.0);

    int expected_nnz = static_cast<int>(n * n * sparsity * 1.5);
    A.values.reserve(expected_nnz);
    A.col_indices.reserve(expected_nnz);
    A.row_ptr.resize(n + 1, 0);

    std::vector<std::vector<std::pair<int, double>>> temp(n);
    temp.reserve(n);

    // Generate strictly diagonally dominant matrix
    for (int i = 0; i < n; ++i) {
        double off_diag_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j && prob_dis(gen) < sparsity) {
                double val = val_dis(gen);
                temp[i].push_back({j, val});
                off_diag_sum += abs(val);
                // Symmetry: add to row j only if j > i to avoid duplicates
                if (j > i) {
                    temp[j].push_back({i, val});
                }
            }
        }
        // Ensure diagonal dominance and positivity
        __PROMISE__ diag_val = off_diag_sum + val_dis(gen) + 1.0; // Positive and dominant
        temp[i].push_back({i, diag_val});
    }

    // Convert to CSR
    for (int i = 0; i < n; ++i) {
        std::sort(temp[i].begin(), temp[i].end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        A.row_ptr[i + 1] = A.row_ptr[i] + temp[i].size();
        for (const auto& entry : temp[i]) {
            A.col_indices.push_back(entry.first);
            A.values.push_back(entry.second);
        }
    }

    std::cout << "Generated matrix: " << n << " x " << n << " with " << A.values.size() << " non-zeros" << std::endl;
    return A;
}

std::vector<__PROMISE__> matvec(const CSRMatrix& A, const std::vector<__PROMISE__>& x) {
    std::vector<__PROMISE__> y(A.n, 0.0);
    const __PROMISE__* __restrict x_data = x.data();
    __PROMISE__* __restrict y_data = y.data();
    const __PROMISE__* __restrict values = A.values.data();
    const int* __restrict col_indices = A.col_indices.data();
    const int* __restrict row_ptr = A.row_ptr.data();

    for (int i = 0; i < A.n; ++i) {
        __PROMISE__ sum = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            sum += values[j] * x_data[col_indices[j]];
        }
        y_data[i] = sum;
    }
    return y;
}

void axpy_inplace(__PROMISE__ alpha, const std::vector<__PROMISE__>& x, std::vector<__PROMISE__>& y) {
    const __PROMISE__* __restrict x_data = x.data();
    __PROMISE__* __restrict y_data = y.data();
    for (size_t i = 0; i < x.size(); ++i) {
        y_data[i] += alpha * x_data[i];
    }
}

__PROMISE__ dot(const std::vector<__PROMISE__>& a, const std::vector<__PROMISE__>& b) {
    __PROMISE__ sum = 0.0;
    const __PROMISE__* __restrict a_data = a.data();
    const __PROMISE__* __restrict b_data = b.data();
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a_data[i] * b_data[i];
    }
    return sum;
}

__PROMISE__ norm(const std::vector<__PROMISE__>& v) {
    __PROMISE__ nrm = dot(v, v);
    if (isnan(nrm) || nrm < 0) return std::numeric_limits<__PROMISE__>::quiet_NaN();
    return sqrt(nrm);
}

struct CGResult {
    std::vector<__PROMISE__> x;
    __PROMISE__ residual;
    int iterations;
};

CGResult conjugate_gradient(const CSRMatrix& A, const std::vector<__PROMISE__>& b, 
                           int max_iter = 100, double tol = 1e-6) {
    int n = A.n;
    std::vector<__PROMISE__> x(n, 0.0);
    std::vector<__PROMISE__> r = b;  // r = b - Ax, initially x = 0 so r = b
    std::vector<__PROMISE__> p = r;
    std::vector<__PROMISE__> Ap(n);
    __PROMISE__ rtr = dot(r, r);
    if (isnan(rtr) || rtr < 0) {
        std::cerr << "Initial residual is invalid: " << rtr << std::endl;
        return {x, std::numeric_limits<__PROMISE__>::quiet_NaN(), 0};
    }
    __PROMISE__ tol2 = tol * tol * dot(b, b);

    int k;
    for (k = 0; k < max_iter && rtr > tol2; ++k) {
        Ap = matvec(A, p);
        __PROMISE__ pAp = dot(p, Ap);
        if (pAp <= 0 || isnan(pAp)) {
            std::cerr << "Matrix not positive definite or numerical error at iteration " << k << ": pAp = " << pAp << std::endl;
            return {x, std::numeric_limits<__PROMISE__>::quiet_NaN(), k};
        }
        __PROMISE__ alpha = rtr / pAp;
        axpy_inplace(alpha, p, x);   // x = x + alpha * p
        axpy_inplace(-alpha, Ap, r); // r = r - alpha * Ap
        __PROMISE__ rtr_new = dot(r, r);
        if (isnan(rtr_new) || isinf(rtr_new)) {
            std::cerr << "Residual became invalid at iteration " << k << ": " << rtr_new << std::endl;
            return {x, std::numeric_limits<__PROMISE__>::quiet_NaN(), k};
        }
        __PROMISE__ beta = rtr_new / rtr;
        if (isnan(beta) || isinf(beta)) {
            std::cerr << "Beta became invalid at iteration " << k << ": " << beta << std::endl;
            return {x, std::numeric_limits<__PROMISE__>::quiet_NaN(), k};
        }
        std::vector<__PROMISE__> p_new = r;
        axpy_inplace(beta, p, p_new); // p_new = r + beta * p
        p = std::move(p_new);
        rtr = rtr_new;
    }

    __PROMISE__ residual = norm(r);
    return {x, residual, k};
}

std::vector<__PROMISE__> generate_rhs(int n) {
    std::vector<__PROMISE__> b(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

int main() {
    int n = 5000;
    CSRMatrix A = generate_random_symmetric_matrix(n, 0.01);
    if (A.n == 0) return 1;

    std::vector<__PROMISE__> b = generate_rhs(A.n);

    auto start = std::chrono::high_resolution_clock::now();
    CGResult result = conjugate_gradient(A, b, 100);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    std::vector<__PROMISE__> Ax = matvec(A, result.x);
    axpy_inplace(-1.0, Ax, b); // b = b - Ax
    __PROMISE__ verify_residual = norm(b);
    std::cout << "Verification residual: " << verify_residual << std::endl;
    // PROMISE_CHECK_ARRAY(result.x.data(), A.n);
    PROMISE_CHECK_VAR(result.residual);
    return 0;
}