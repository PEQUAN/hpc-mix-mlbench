#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

const double EPS = 1e-10; // Tolerance for convergence
const int MAX_ITER = 1000; // Maximum QR iterations


struct CSRMatrix {
    int n;
    double* values;
    int* col_indices;
    int* row_ptr;
    int nnz;
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

CSRMatrix generate_random_matrix(int n, double sparsity = 0.01) {
    CSRMatrix A = {n, nullptr, nullptr, nullptr, 0};
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    std::uniform_real_distribution<> prob(0.0, 1.0);

    Pair** temp = new Pair*[n]();
    int* temp_sizes = new int[n]();
    for (int i = 0; i < n; ++i) {
        temp[i] = new Pair[n]();
    }

    for (int i = 0; i < n; ++i) {
        temp[i][temp_sizes[i]] = {i, 0.0};
        temp_sizes[i]++;
        for (int j = 0; j < n; ++j) {
            if (i != j && prob(gen) < sparsity) {
                temp[i][temp_sizes[i]] = {j, dis(gen)};
                temp_sizes[i]++;
                if (i < j) {
                    temp[j][temp_sizes[j]] = {i, temp[i][temp_sizes[i]-1].second};
                    temp_sizes[j]++;
                }
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        double off_diag_sum = 0.0;
        for (int k = 0; k < temp_sizes[i]; ++k) {
            if (temp[i][k].first != i) {
                off_diag_sum += std::abs(temp[i][k].second);
            }
        }
        for (int k = 0; k < temp_sizes[i]; ++k) {
            if (temp[i][k].first == i) {
                temp[i][k].second = off_diag_sum + 1.0;
                break;
            }
        }
    }

    A.nnz = 0;
    for (int i = 0; i < n; ++i) {
        A.nnz += temp_sizes[i];
    }

    A.values = new double[A.nnz];
    A.col_indices = new int[A.nnz];
    A.row_ptr = new int[n + 1];
    A.row_ptr[0] = 0;

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

    for (int i = 0; i < n; ++i) delete[] temp[i];
    delete[] temp;
    delete[] temp_sizes;

    std::cout << "Generated matrix: " << n << " x " << n << " with " << A.nnz << " non-zeros" << std::endl;
    return A;
}

void csr_to_dense(int n, const CSRMatrix& A, double* dense) {
    std::memset(dense, 0, n * n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        for (int idx = A.row_ptr[i]; idx < A.row_ptr[i + 1]; ++idx) {
            int j = A.col_indices[idx];
            dense[i * n + j] = A.values[idx];
        }
    }
}

void copy_matrix(int n, const double* A, double* B) {
    std::memcpy(B, A, n * n * sizeof(double));
}

void zero_matrix(int n, double* A) {
    std::memset(A, 0, n * n * sizeof(double));
}

double frobenius_norm(int n, const double* A) {
    double sum = 0.0;
    for (int i = 0; i < n * n; i++)
        sum += A[i] * A[i];
    return std::sqrt(sum);
}

void matrix_mult(int n, const double* A, const double* B, double* C) {
    zero_matrix(n, C);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i*n + j] += A[i*n + k] * B[k*n + j];
}

void householder(int n, int k, double* A) {
    double* v = new double[n];
    double norm = 0.0;
    for (int i = k + 1; i < n; i++) {
        v[i] = A[i*n + k];
        norm += v[i] * v[i];
    }
    norm = std::sqrt(norm);
    if (norm < EPS) {
        delete[] v;
        return;
    }

    v[k + 1] += (A[(k + 1)*n + k] > 0 ? norm : -norm);
    norm = std::sqrt(norm * norm + v[k + 1] * v[k + 1]);
    for (int i = k + 1; i < n; i++) v[i] /= norm;

    double* P = new double[n * n];
    double* temp = new double[n * n];
    zero_matrix(n, P);
    for (int i = 0; i < n; i++) P[i*n + i] = 1.0;
    for (int i = k + 1; i < n; i++)
        for (int j = k + 1; j < n; j++)
            P[i*n + j] -= 2.0 * v[i] * v[j];

    matrix_mult(n, P, A, temp);
    matrix_mult(n, temp, P, A);

    delete[] v;
    delete[] P;
    delete[] temp;
}

void to_hessenberg(int n, double* A) {
    for (int k = 0; k < n - 2; k++) {
        householder(n, k, A);
        for (int i = k + 2; i < n; i++)
            A[i*n + k] = 0.0;
    }
}

void givens_rotation(int n, int i, double a, double b, double& c, double& s) {
    double r = std::sqrt(a * a + b * b);
    if (r < EPS) { c = 1.0; s = 0.0; return; }
    c = a / r;
    s = b / r;
}

void qr_decomposition(int n, const double* A, double* Q, double* R) {
    copy_matrix(n, A, R);
    zero_matrix(n, Q);
    for (int i = 0; i < n; i++) Q[i*n + i] = 1.0;

    for (int j = 0; j < n - 1; j++) {
        for (int i = j + 1; i < n; i++) {
            double c, s;
            givens_rotation(n, i, R[j*n + j], R[i*n + j], c, s);
            for (int k = j; k < n; k++) {
                double t1 = c * R[j*n + k] + s * R[i*n + k];
                double t2 = -s * R[j*n + k] + c * R[i*n + k];
                R[j*n + k] = t1;
                R[i*n + k] = t2;
            }
            for (int k = 0; k < n; k++) {
                double t1 = c * Q[k*n + j] + s * Q[k*n + i];
                double t2 = -s * Q[k*n + j] + c * Q[k*n + i];
                Q[k*n + j] = t1;
                Q[k*n + i] = t2;
            }
            R[i*n + j] = 0.0;
        }
    }
}

void inverse_iteration(int n, const double* A, double lambda, double* v) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    for (int i = 0; i < n; i++) v[i] = dis(gen);
    double norm = 0.0;
    for (int i = 0; i < n; i++) norm += v[i] * v[i];
    norm = std::sqrt(norm);
    for (int i = 0; i < n; i++) v[i] /= norm;

    for (int iter = 0; iter < 5; iter++) {
        double* b = new double[n];
        std::memcpy(b, v, n * sizeof(double));
        double* temp_v = new double[n]();

        double* M = new double[n * n];
        copy_matrix(n, A, M);
        for (int i = 0; i < n; i++) M[i*n + i] -= lambda;

        for (int k = 0; k < n - 1; k++) {
            for (int i = k + 1; i < n; i++) {
                if (std::abs(M[k*n + k]) < EPS) continue;
                double factor = M[i*n + k] / M[k*n + k];
                for (int j = k; j < n; j++)
                    M[i*n + j] -= factor * M[k*n + j];
                b[i] -= factor * b[k];
            }
        }

        for (int i = n - 1; i >= 0; i--) {
            double sum = b[i];
            for (int j = i + 1; j < n; j++)
                sum -= M[i*n + j] * temp_v[j];
            temp_v[i] = sum / M[i*n + i];
        }

        norm = 0.0;
        for (int i = 0; i < n; i++) norm += temp_v[i] * temp_v[i];
        norm = std::sqrt(norm);
        for (int i = 0; i < n; i++) v[i] = temp_v[i] / norm;

        delete[] b;
        delete[] temp_v;
        delete[] M;
    }
}

void qr_algorithm(int n, double* A, double* eigenvalues) {
    double* Ak = new double[n * n];
    double* Q = new double[n * n];
    double* R = new double[n * n];
    double* temp = new double[n * n];
    copy_matrix(n, A, Ak);
    to_hessenberg(n, Ak);

    int iter = 0;
    while (iter < MAX_ITER) {
        double a = Ak[(n-2)*n + n-2], b = Ak[(n-2)*n + n-1];
        double c = Ak[(n-1)*n + n-2], d = Ak[(n-1)*n + n-1];
        double trace = a + d, det = a * d - b * c;
        double disc = std::sqrt(trace * trace - 4 * det);
        double sigma = (trace + (trace > 0 ? disc : -disc)) / 2.0;

        for (int i = 0; i < n; i++) Ak[i*n + i] -= sigma;
        qr_decomposition(n, Ak, Q, R);
        matrix_mult(n, R, Q, Ak);
        for (int i = 0; i < n; i++) Ak[i*n + i] += sigma;

        bool converged = true;
        for (int i = 1; i < n; i++) {
            if (std::abs(Ak[i*n + i-1]) > EPS) {
                converged = false;
                break;
            }
        }
        if (converged) break;
        iter++;
    }

    for (int i = 0; i < n; i++) eigenvalues[i] = Ak[i*n + i];

    delete[] Ak;
    delete[] Q;
    delete[] R;
    delete[] temp;
}

double evaluate_qr_factorization(int n, const double* A, const double* Q, const double* R) {
// Evaluate QR factorization: ||A - QR||_F
    double* QR = new double[n * n];
    double* residual = new double[n * n];
    matrix_mult(n, Q, R, QR);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            residual[i*n + j] = A[i*n + j] - QR[i*n + j];
    double norm = frobenius_norm(n, residual);
    delete[] QR;
    delete[] residual;
    return norm;
}

double compute_eigen_residual(int n, const double* A, double lambda, const double* v) {
    double* Av = new double[n]();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            Av[i] += A[i*n + j] * v[j];
    double* residual = new double[n];
    for (int i = 0; i < n; i++)
        residual[i] = Av[i] - lambda * v[i];
    double norm = 0.0;
    for (int i = 0; i < n; i++)
        norm += residual[i] * residual[i];
    norm = std::sqrt(norm);
    delete[] Av;
    delete[] residual;
    return norm;
}

int main() {
    int n = 100; 
    double sparsity = 0.01;

    CSRMatrix A = generate_random_matrix(n, sparsity);

    double* dense_A = new double[n * n];
    csr_to_dense(n, A, dense_A);

    double* eigenvalues = new double[n];
    qr_algorithm(n, dense_A, eigenvalues);


    double* Q = new double[n * n];
    double* R = new double[n * n];
    qr_decomposition(n, dense_A, Q, R);
    double qr_residual = evaluate_qr_factorization(n, dense_A, Q, R);
    std::cout << "QR factorization residual (||A - QR||_F): " << qr_residual << "\n";

    // Evaluate one eigenvalue residual
    // double* v = new double[n];
    // inverse_iteration(n, dense_A, eigenvalues[0], v);
    // double eigen_residual = compute_eigen_residual(n, dense_A, eigenvalues[0], v);
    // std::cout << "Residual for first eigenvalue (||Av - lambda v||_2): " << eigen_residual << "\n";

    free_csr_matrix(A);
    delete[] dense_A;
    delete[] eigenvalues;
    delete[] Q;
    delete[] R;
    // delete[] v;

    return 0;
}