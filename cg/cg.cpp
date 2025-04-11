#ifndef _Alignof
#define _Alignof(type) alignof(type)
#endif

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
    std::vector<double> values;
    std::vector<int> col_indices;
    std::vector<int> row_ptr;
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

    std::vector<std::vector<std::pair<int, double>>> temp(n);
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

std::vector<double> matvec(const CSRMatrix& A, const std::vector<double>& x) {
    std::vector<double> y(A.n, 0.0);
    for (int i = 0; i < A.n; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            y[i] += A.values[j] * x[A.col_indices[j]];
        }
    }
    return y;
}


std::vector<double> axpy(double alpha, const std::vector<double>& x, const std::vector<double>& y) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = alpha * x[i] + y[i];
    }
    return result;
}


double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double norm(const std::vector<double>& v) {
    return std::sqrt(dot(v, v));
}

struct CGResult {
    std::vector<double> x;
    double residual;
    int iterations;
};

CGResult conjugate_gradient(const CSRMatrix& A, const std::vector<double>& b, 
                           int max_iter = 1000, double tol = 1e-6) {
    int n = A.n;
    std::vector<double> x(n, 0.0);
    std::vector<double> r = b;
    std::vector<double> p = r;
    double rtr = dot(r, r);
    double tol2 = tol * tol * dot(b, b);

    int k;
    for (k = 0; k < max_iter && rtr > tol2; ++k) {
        std::vector<double> Ap = matvec(A, p);
        double alpha = rtr / dot(p, Ap);
        x = axpy(alpha, p, x);
        r = axpy(-alpha, Ap, r);
        double rtr_new = dot(r, r);
        double beta = rtr_new / rtr;
        p = axpy(beta, p, r);
        rtr = rtr_new;
    }

    double residual = norm(r);
    return {x, residual, k};
}

std::vector<double> generate_rhs(int n) {
    std::vector<double> b(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(1.0, 10.0);
    for (int i = 0; i < n; ++i) {
        b[i] = dis(gen);
    }
    return b;
}

void write_solution(const std::vector<double>& x, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening output file" << std::endl;
        return;
    }
    file << "x\n";
    for (double val : x) {
        file << val << "\n";
    }
}

int main() {
    std::string filename = "../data/suitesparse/rdb5000.mtx";
    CSRMatrix A = read_mtx_file(filename);
    if (A.n == 0) return 1;

    std::vector<double> b = generate_rhs(A.n);

    auto start = std::chrono::high_resolution_clock::now();
    CGResult result = conjugate_gradient(A, b, A.n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Matrix size: " << A.n << " x " << A.n << std::endl;
    std::cout << "Training time: " << duration.count() << " ms" << std::endl;
    std::cout << "Final residual: " << result.residual << std::endl;
    std::cout << "Iterations to converge: " << result.iterations << std::endl;

    std::vector<double> Ax = matvec(A, result.x);
    double verify_residual = norm(axpy(-1.0, Ax, b));
    std::cout << "Verification residual: " << verify_residual << std::endl;

    write_solution(result.x, "../results/cg/cg_solution.csv");

    return 0;
}