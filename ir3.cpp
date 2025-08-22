
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>

template<typename T>
T* allocate_matrix(int rows, int cols) {
    return new T[rows * cols]();
}

template<typename T>
void deallocate_matrix(T* matrix) {
    delete[] matrix;
}

template<typename T>
T get_element(T* matrix, int rows, int cols, int i, int j) {
    return matrix[i * cols + j];
}

template<typename T>
void set_element(T* matrix, int rows, int cols, int i, int j, T value) {
    matrix[i * cols + j] = value;
}

template<typename T>
T* matrix_multiply(T* A, int rowsA, int colsA, T* B, int rowsB, int colsB) {
    if (colsA != rowsB) {
        std::cerr << "Matrix dimensions incompatible\n";
        return nullptr;
    }
    T* C = allocate_matrix<T>(rowsA, colsB);
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            T sum = T(0);
            for (int k = 0; k < colsA; ++k) {
                sum += get_element(A, rowsA, colsA, i, k) * get_element(B, rowsB, colsB, k, j);
            }
            set_element(C, rowsA, colsB, i, j, sum);
        }
    }
    return C;
}

template<typename T>
T* transpose(T* A, int rows, int cols) {
    T* T_mat = allocate_matrix<T>(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            set_element(T_mat, cols, rows, j, i, get_element(A, rows, cols, i, j));
        }
    }
    return T_mat;
}

template<typename T>
T* gram_schmidt(T* A, int n) {
    T* Q = allocate_matrix<T>(n, n);
    T* temp = new T[n];
    
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            temp[i] = get_element(A, n, n, i, j);
        }
        for (int k = 0; k < j; ++k) {
            T dot = T(0);
            for (int i = 0; i < n; ++i) {
                dot += get_element(Q, n, n, i, k) * get_element(A, n, n, i, j);
            }
            for (int i = 0; i < n; ++i) {
                temp[i] -= dot * get_element(Q, n, n, i, k);
            }
        }
        T norm = T(0);
        for (int i = 0; i < n; ++i) {
            norm += temp[i] * temp[i];
        }
        norm = std::sqrt(norm);
        if (norm < T(1e-10)) norm = T(1e-10);
        for (int i = 0; i < n; ++i) {
            set_element(Q, n, n, i, j, temp[i] / norm);
        }
    }
    delete[] temp;
    return Q;
}

double compute_condition_number(double* sigma, int p, int kl, int ku, int m, int n) {
    if (kl < m - 1 || ku < n - 1) {
        std::cout << "Warning: Condition number for banded matrix may not be exact.\n";
    }
    double sigma_max = sigma[0];
    double sigma_min = sigma[p - 1];
    for (int i = 1; i < p; ++i) {
        if (sigma[i] > sigma_max) sigma_max = sigma[i];
        if (sigma[i] < sigma_min && sigma[i] > 0) sigma_min = sigma[i];
    }
    return sigma_max / sigma_min;
}

double* gallery_randsvd(int n, double kappa, int mode = 3, int kl = -1, int ku = -1, int method = 0, int random_state = 42) {
    std::mt19937 rng(random_state);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    if (n < 1) {
        std::cerr << "n must be a positive integer\n";
        return nullptr;
    }
    int m = n;
    
    if (kappa < 0) {
        kappa = std::sqrt(1.0 / std::numeric_limits<double>::epsilon());
    }
    
    if (kl < 0) kl = m - 1;
    if (ku < 0) ku = kl;
    if (kl >= m || ku >= n || kl < 0 || ku < 0) {
        std::cerr << "kl and ku must be non-negative and less than matrix dimensions\n";
        return nullptr;
    }
    
    if (mode < -5 || mode > 5 || mode == 0) {
        std::cerr << "Mode must be an integer from -5 to -1 or 1 to 5\n";
        return nullptr;
    }
    if (method != 0 && method != 1) {
        std::cerr << "Method must be 0 or 1\n";
        return nullptr;
    }
    if (kl < m - 1 || ku < n - 1) {
        std::cout << "Warning: Banded matrix may not preserve exact singular values.\n";
    }
    
    int p = std::min(m, n);
    double* sigma = new double[p]();
    
    if (kappa <= 1) {
        if (m != n) {
            std::cerr << "For kappa <= 1, matrix must be square (m == n)\n";
            delete[] sigma;
            return nullptr;
        }
        double lambda_min = std::abs(kappa);
        double lambda_max = 1.0;
        
        if (mode == 1 || mode == -1) {
            for (int i = 0; i < p; ++i) sigma[i] = lambda_min;
            sigma[0] = lambda_max;
        } else if (mode == 2 || mode == -2) {
            for (int i = 0; i < p; ++i) sigma[i] = lambda_max;
            sigma[p-1] = lambda_min;
        } else if (mode == 3 || mode == -3) {
            for (int k = 0; k < p; ++k) {
                sigma[k] = lambda_max * std::pow(lambda_min / lambda_max, k / (p > 1 ? p - 1.0 : 1.0));
            }
        } else if (mode == 4 || mode == -4) {
            for (int k = 0; k < p; ++k) {
                sigma[k] = lambda_max - (k / (p > 1 ? p - 1.0 : 1.0)) * (lambda_max - lambda_min);
            }
        } else if (mode == 5 || mode == -5) {
            std::uniform_real_distribution<double> unif(0.0, 1.0);
            sigma[0] = lambda_max;
            if (p > 1) sigma[p-1] = lambda_min;
            for (int i = 1; i < p-1; ++i) {
                double r = unif(rng);
                sigma[i] = lambda_max * std::exp(std::log(lambda_min / lambda_max) * r);
            }
        }
        
        if (mode < 0) {
            std::sort(sigma, sigma + p);
        } else {
            std::sort(sigma, sigma + p, std::greater<double>());
        }
        
        for (int i = 0; i < p; ++i) {
            if (sigma[i] <= 0) {
                std::cerr << "Eigenvalues must be positive for symmetric positive definite matrix\n";
                delete[] sigma;
                return nullptr;
            }
        }
        
        double* X = allocate_matrix<double>(n, n);
        for (int i = 0; i < n * n; ++i) {
            X[i] = dist(rng);
        }
        
        double* Q = gram_schmidt(X, n);
        deallocate_matrix(X);
        
        double* D = allocate_matrix<double>(n, n);
        for (int i = 0; i < n; ++i) {
            set_element(D, n, n, i, i, sigma[i]);
        }
        
        double* QT = transpose(Q, n, n);
        double* temp = matrix_multiply(D, n, n, QT, n, n);
        double* A = matrix_multiply(Q, n, n, temp, n, n);
        
        std::cout << "Condition number: " << compute_condition_number(sigma, p, kl, ku, n, n) << "\n";
        
        deallocate_matrix(Q);
        deallocate_matrix(D);
        deallocate_matrix(QT);
        deallocate_matrix(temp);
        delete[] sigma;
        return A;
    }
    
    if (std::abs(kappa) < 1) {
        std::cerr << "For non-symmetric case, abs(kappa) must be >= 1\n";
        delete[] sigma;
        return nullptr;
    }
    
    double sigma_max = 1.0;
    double sigma_min = sigma_max / std::abs(kappa);
    
    if (mode == 1 || mode == -1) {
        for (int i = 0; i < p; ++i) sigma[i] = sigma_min;
        sigma[0] = sigma_max;
    } else if (mode == 2 || mode == -2) {
        for (int i = 0; i < p; ++i) sigma[i] = sigma_max;
        sigma[p-1] = sigma_min;
    } else if (mode == 3 || mode == -3) {
        for (int k = 0; k < p; ++k) {
            sigma[k] = sigma_max * std::pow(sigma_min / sigma_max, k / (p > 1 ? p - 1.0 : 1.0));
        }
    } else if (mode == 4 || mode == -4) {
        for (int k = 0; k < p; ++k) {
            sigma[k] = sigma_max - (k / (p > 1 ? p - 1.0 : 1.0)) * (sigma_max - sigma_min);
        }
    } else if (mode == 5 || mode == -5) {
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        sigma[0] = sigma_max;
        if (p > 1) sigma[p-1] = sigma_min;
        for (int i = 1; i < p-1; ++i) {
            double r = unif(rng);
            sigma[i] = sigma_max * std::exp(std::log(sigma_min / sigma_max) * r);
        }
    }
    
    if (mode < 0) {
        std::sort(sigma, sigma + p);
    } else {
        std::sort(sigma, sigma + p, std::greater<double>());
    }
    
    std::cout << "Generated sigma for mode=" << mode << ": ";
    for (int i = 0; i < p; ++i) {
        std::cout << sigma[i] << " ";
    }
    std::cout << "\n";
    
    double* Sigma = allocate_matrix<double>(m, n);
    for (int i = 0; i < p; ++i) {
        set_element(Sigma, m, n, i, i, sigma[i]);
    }
    
    double* X = allocate_matrix<double>(m, m);
    double* Y = allocate_matrix<double>(n, n);
    for (int i = 0; i < m * m; ++i) {
        X[i] = dist(rng);
    }
    for (int i = 0; i < n * n; ++i) {
        Y[i] = dist(rng);
    }
    
    double* U = gram_schmidt(X, m);
    double* V = gram_schmidt(Y, n);
    deallocate_matrix(X);
    deallocate_matrix(Y);
    
    double* VT = transpose(V, n, n);
    double* temp = matrix_multiply(Sigma, m, n, VT, n, n);
    double* A = matrix_multiply(U, m, m, temp, m, n);
    
    if (kl < m - 1 || ku < n - 1) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (j < i - kl || j > i + ku) {
                    set_element(A, m, n, i, j, 0.0);
                }
            }
        }
    }
    
    std::cout << "Condition number: " << compute_condition_number(sigma, p, kl, ku, m, n) << "\n";
    
    deallocate_matrix(U);
    deallocate_matrix(V);
    deallocate_matrix(VT);
    deallocate_matrix(Sigma);
    deallocate_matrix(temp);
    delete[] sigma;
    
    return A;
}

// LU factorization with partial pivoting
template<typename T>
void lu_factorization(T* A, T* L, T* U, int* P, int n) {
    for (int i = 0; i < n; ++i) {
        P[i] = i;
        set_element(L, n, n, i, i, T(1));
    }
    
    for (int k = 0; k < n; ++k) {
        int max_idx = k;
        T max_val = std::abs(get_element(A, n, n, k, k));
        for (int i = k + 1; i < n; ++i) {
            if (std::abs(get_element(A, n, n, i, k)) > max_val) {
                max_val = std::abs(get_element(A, n, n, i, k));
                max_idx = i;
            }
        }
        if (max_idx != k) {
            for (int j = 0; j < n; ++j) {
                T temp = get_element(A, n, n, k, j);
                set_element(A, n, n, k, j, get_element(A, n, n, max_idx, j));
                set_element(A, n, n, max_idx, j, temp);
            }
            int temp = P[k];
            P[k] = P[max_idx];
            P[max_idx] = temp;
        }
        if (std::abs(get_element(A, n, n, k, k)) < T(1e-10)) {
            std::cerr << "Matrix is singular\n";
            return;
        }
        for (int i = k + 1; i < n; ++i) {
            T factor = get_element(A, n, n, i, k) / get_element(A, n, n, k, k);
            set_element(L, n, n, i, k, factor);
            for (int j = k; j < n; ++j) {
                set_element(A, n, n, i, j, get_element(A, n, n, i, j) - factor * get_element(A, n, n, k, j));
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i <= j) {
                set_element(U, n, n, i, j, get_element(A, n, n, i, j));
            }
        }
    }
}

// Forward and backward substitution with mixed precision
template<typename T, typename T_high>
void forward_substitution(T* L, T_high* b, T_high* y, int n) {
    for (int i = 0; i < n; ++i) {
        T_high sum = b[i];
        for (int j = 0; j < i; ++j) {
            sum -= static_cast<T_high>(get_element(L, n, n, i, j)) * y[j];
        }
        y[i] = sum / static_cast<T_high>(get_element(L, n, n, i, i));
    }
}

template<typename T, typename T_high>
void backward_substitution(T* U, T_high* y, T_high* x, int n) {
    for (int i = n - 1; i >= 0; --i) {
        T_high sum = y[i];
        for (int j = i + 1; j < n; ++j) {
            sum -= static_cast<T_high>(get_element(U, n, n, i, j)) * x[j];
        }
        x[i] = sum / static_cast<T_high>(get_element(U, n, n, i, i));
    }
}

// Compute Givens rotation parameters
template<typename T>
void rotmat(T a, T b, T& c, T& s) {
    if (b == T(0)) {
        c = T(1);
        s = T(0);
    } else if (std::abs(b) > std::abs(a)) {
        T temp = a / b;
        s = T(1) / std::sqrt(T(1) + temp * temp);
        c = temp * s;
    } else {
        T temp = b / a;
        c = T(1) / std::sqrt(T(1) + temp * temp);
        s = temp * c;
    }
}

// GMRES implementation following MATLAB gmres_dq
template<typename Tf, typename Tr, typename Tg>
void gmres(Tf* A, Tg* x, Tr* b, Tf* L, Tf* U, int* P, int restrt, int max_iter, double tol, int* gmres_iter, double* gmres_err) {
    // Initialize workspace
    int n = restrt; // Assuming square matrix
    double* V = allocate_matrix<double>(n, restrt + 1); // Use double for V
    double* H = allocate_matrix<double>(restrt + 1, restrt); // Use double for H
    double* cs = new double[restrt]();
    double* sn = new double[restrt]();
    double* s = new double[restrt + 1]();
    double* e1 = new double[restrt + 1]();
    e1[0] = 1.0;

    // Compute initial residual: r = U^-1 L^-1 (b - Ax)
    Tr* rtmp = allocate_matrix<Tr>(n, 1);
    for (int i = 0; i < n; ++i) {
        Tr sum = b[i];
        for (int j = 0; j < n; ++j) {
            sum -= static_cast<Tr>(get_element(A, n, n, i, j)) * static_cast<Tr>(x[j]);
        }
        rtmp[i] = sum;
    }
    Tr* r_high = allocate_matrix<Tr>(n, 1);
    Tr* y_high = allocate_matrix<Tr>(n, 1);
    if (L != nullptr && U != nullptr && P != nullptr) {
        for (int i = 0; i < n; ++i) r_high[P[i]] = rtmp[i];
        forward_substitution(L, r_high, y_high, n);
        backward_substitution(U, y_high, r_high, n);
    } else {
        for (int i = 0; i < n; ++i) r_high[i] = rtmp[i];
    }
    double* r = allocate_matrix<double>(n, 1); // Use double for r
    for (int i = 0; i < n; ++i) r[i] = static_cast<double>(r_high[i]);

    // Compute initial residual norm
    double bnrm2 = 0.0;
    for (int i = 0; i < n; ++i) bnrm2 += r[i] * r[i];
    bnrm2 = std::sqrt(bnrm2);
    if (bnrm2 == 0.0) bnrm2 = 1.0;
    *gmres_err = bnrm2 / bnrm2; // Relative residual norm
    *gmres_iter = 0;

    if (*gmres_err <= tol) {
        deallocate_matrix(V);
        deallocate_matrix(H);
        delete[] cs;
        delete[] sn;
        delete[] s;
        delete[] e1;
        deallocate_matrix(rtmp);
        deallocate_matrix(r_high);
        deallocate_matrix(y_high);
        deallocate_matrix(r);
        return;
    }

    int iter = 0;
    while (iter < max_iter) {
        // Initialize V(:,1) = r / norm(r)
        double r_norm = 0.0;
        for (int i = 0; i < n; ++i) r_norm += r[i] * r[i];
        r_norm = std::sqrt(r_norm);
        if (r_norm == 0.0) r_norm = 1e-12;
        for (int i = 0; i < n; ++i) {
            set_element(V, n, restrt + 1, i, 0, r[i] / r_norm);
        }

        // Initialize s = norm(r) * e1
        for (int i = 0; i < restrt + 1; ++i) s[i] = 0.0;
        s[0] = r_norm;

        // Arnoldi process
        for (int i = 0; i < restrt && iter < max_iter; ++i) {
            (*gmres_iter)++;
            iter++;

            // w = U^-1 L^-1 A V(:,i)
            Tr* vcur = allocate_matrix<Tr>(n, 1);
            for (int j = 0; j < n; ++j) vcur[j] = static_cast<Tr>(get_element(V, n, restrt + 1, j, i));
            Tr* Av = allocate_matrix<Tr>(n, 1);
            for (int j = 0; j < n; ++j) {
                Tr sum = Tr(0);
                for (int k = 0; k < n; ++k) {
                    sum += static_cast<Tr>(get_element(A, n, n, j, k)) * vcur[k];
                }
                Av[j] = sum;
            }
            if (L != nullptr && U != nullptr && P != nullptr) {
                for (int j = 0; j < n; ++j) vcur[P[j]] = Av[j];
                forward_substitution(L, vcur, y_high, n);
                backward_substitution(U, y_high, Av, n);
            } else {
                for (int j = 0; j < n; ++j) vcur[j] = Av[j];
            }
            double* w = allocate_matrix<double>(n, 1); // Use double for w
            for (int j = 0; j < n; ++j) w[j] = static_cast<double>(Av[j]);

            // Classical Gram-Schmidt
            for (int k = 0; k <= i; ++k) {
                double h = 0.0;
                for (int j = 0; j < n; ++j) {
                    h += w[j] * get_element(V, n, restrt + 1, j, k);
                }
                set_element(H, restrt + 1, restrt, k, i, h);
                for (int j = 0; j < n; ++j) {
                    w[j] -= h * get_element(V, n, restrt + 1, j, k);
                }
            }
            double h_norm = 0.0;
            for (int j = 0; j < n; ++j) h_norm += w[j] * w[j];
            h_norm = std::sqrt(h_norm);
            set_element(H, restrt + 1, restrt, i + 1, i, h_norm);
            if (h_norm > 1e-12) {
                for (int j = 0; j < n; ++j) {
                    set_element(V, n, restrt + 1, j, i + 1, w[j] / h_norm);
                }
            }

            // Apply Givens rotations
            for (int k = 0; k < i; ++k) {
                double temp = cs[k] * get_element(H, restrt + 1, restrt, k, i) +
                              sn[k] * get_element(H, restrt + 1, restrt, k + 1, i);
                set_element(H, restrt + 1, restrt, k + 1, i,
                            -sn[k] * get_element(H, restrt + 1, restrt, k, i) +
                             cs[k] * get_element(H, restrt + 1, restrt, k + 1, i));
                set_element(H, restrt + 1, restrt, k, i, temp);
            }
            double c, s_val;
            rotmat(get_element(H, restrt + 1, restrt, i, i), get_element(H, restrt + 1, restrt, i + 1, i), c, s_val);
            cs[i] = c;
            sn[i] = s_val;
            double temp = c * s[i];
            s[i + 1] = -s_val * s[i];
            s[i] = temp;
            set_element(H, restrt + 1, restrt, i, i, c * get_element(H, restrt + 1, restrt, i, i) +
                    s_val * get_element(H, restrt + 1, restrt, i + 1, i));
            set_element(H, restrt + 1, restrt, i + 1, i, 0.0);

            // Check convergence
            *gmres_err = std::abs(s[i + 1]) / bnrm2;
            if (*gmres_err <= tol || iter >= max_iter) {
                // Solve for y: H(1:i,1:i) y = s(1:i)
                double* y = allocate_matrix<double>(i + 1, 1);
                for (int j = 0; j <= i; ++j) y[j] = s[j];
                for (int j = 0; j <= i; ++j) {
                    double pivot = get_element(H, restrt + 1, restrt, j, j);
                    if (std::abs(pivot) < 1e-12) break;
                    for (int m = j + 1; m <= i; ++m) {
                        double factor = get_element(H, restrt + 1, restrt, m, j) / pivot;
                        for (int p = j; p <= i; ++p) {
                            set_element(H, restrt + 1, restrt, m, p,
                                        get_element(H, restrt + 1, restrt, m, p) - factor * get_element(H, restrt + 1, restrt, j, p));
                        }
                        y[m] -= factor * y[j];
                    }
                }
                for (int j = i; j >= 0; --j) {
                    double sum = y[j];
                    for (int m = j + 1; m <= i; ++m) {
                        sum -= get_element(H, restrt + 1, restrt, j, m) * y[m];
                    }
                    y[j] = sum / get_element(H, restrt + 1, restrt, j, j);
                }

                // Update x = x + V(:,1:i+1) * y
                for (int j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (int m = 0; m <= i; ++m) {
                        sum += get_element(V, n, restrt + 1, j, m) * y[m];
                    }
                    x[j] += static_cast<Tg>(sum);
                }
                deallocate_matrix(y);

                if (*gmres_err <= tol) {
                    deallocate_matrix(V);
                    deallocate_matrix(H);
                    delete[] cs;
                    delete[] sn;
                    delete[] s;
                    delete[] e1;
                    deallocate_matrix(rtmp);
                    deallocate_matrix(r_high);
                    deallocate_matrix(y_high);
                    deallocate_matrix(r);
                    deallocate_matrix(vcur);
                    deallocate_matrix(Av);
                    deallocate_matrix(w);
                    return;
                }
            }

            deallocate_matrix(vcur);
            deallocate_matrix(Av);
            deallocate_matrix(w);
        }

        // Solve for y: H(1:m,1:m) \ s(1:m)
        double* y = allocate_matrix<double>(restrt, 1);
        for (int j = 0; j < restrt; ++j) y[j] = s[j];
        for (int j = 0; j < restrt; ++j) {
            double pivot = get_element(H, restrt + 1, restrt, j, j);
            if (std::abs(pivot) < 1e-12) break;
            for (int m = j + 1; m < restrt; ++m) {
                double factor = get_element(H, restrt + 1, restrt, m, j) / pivot;
                for (int p = j; p < restrt; ++p) {
                    set_element(H, restrt + 1, restrt, m, p,
                                get_element(H, restrt + 1, restrt, m, p) - factor * get_element(H, restrt + 1, restrt, j, p));
                }
                y[m] -= factor * y[j];
            }
        }
        for (int j = restrt - 1; j >= 0; --j) {
            double sum = y[j];
            for (int m = j + 1; m < restrt; ++m) {
                sum -= get_element(H, restrt + 1, restrt, j, m) * y[m];
            }
            y[j] = sum / get_element(H, restrt + 1, restrt, j, j);
        }

        // Update x = x + V(:,1:m) * y
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int m = 0; m < restrt; ++m) {
                sum += get_element(V, n, restrt + 1, j, m) * y[m];
            }
            x[j] += static_cast<Tg>(sum);
        }
        deallocate_matrix(y);

        // Compute new residual
        for (int i = 0; i < n; ++i) {
            Tr sum = b[i];
            for (int j = 0; j < n; ++j) {
                sum -= static_cast<Tr>(get_element(A, n, n, i, j)) * static_cast<Tr>(x[j]);
            }
            rtmp[i] = sum;
        }
        if (L != nullptr && U != nullptr && P != nullptr) {
            for (int i = 0; i < n; ++i) r_high[P[i]] = rtmp[i];
            forward_substitution(L, r_high, y_high, n);
            backward_substitution(U, y_high, r_high, n);
        } else {
            for (int i = 0; i < n; ++i) r_high[i] = rtmp[i];
        }
        for (int i = 0; i < n; ++i) r[i] = static_cast<double>(r_high[i]);

        double r_norm_new = 0.0;
        for (int i = 0; i < n; ++i) r_norm_new += r[i] * r[i];
        r_norm_new = std::sqrt(r_norm_new);
        *gmres_err = r_norm_new / bnrm2;
        if (*gmres_err <= tol) {
            deallocate_matrix(V);
            deallocate_matrix(H);
            delete[] cs;
            delete[] sn;
            delete[] s;
            delete[] e1;
            deallocate_matrix(rtmp);
            deallocate_matrix(r_high);
            deallocate_matrix(y_high);
            deallocate_matrix(r);
            return;
        }
    }

    if (*gmres_err > tol) *gmres_iter = -1; // Indicate non-convergence

    deallocate_matrix(V);
    deallocate_matrix(H);
    delete[] cs;
    delete[] sn;
    delete[] s;
    delete[] e1;
    deallocate_matrix(rtmp);
    deallocate_matrix(r_high);
    deallocate_matrix(y_high);
    deallocate_matrix(r);
}

// GMRES-IR with mixed precisions
template<typename Tf, typename Tr, typename Tg, typename Tu>
void gmres_ir(double* A, double* b, double* x_true, int n, double kappa, int precf, int precw, int precr, int max_iter, double gtol) {
    // Validate precision inputs
    if (precf != 1 && precf != 2) {
        std::cerr << "precf should be 1 (single) or 2 (double)\n";
        return;
    }
    if (precw != 1 && precw != 2 && precw != 3) {
        std::cerr << "precw should be 1 (single), 2 (double), or 3 (long double)\n";
        return;
    }
    if (precr != 1 && precr != 2) {
        std::cerr << "precr should be 1 (single) or 2 (long double)\n";
        return;
    }

    // Set machine epsilon for working precision
    double u = (precw == 1) ? 1e-7 : (precw == 2 ? 1e-16 : 1e-19);

    // Print precision settings
    std::cout << "\nGMRES-IR (kappa=" << kappa << ", uf=" << (precf == 1 ? "single" : "double") 
              << ", u=" << (precw == 1 ? "single" : (precw == 2 ? "double" : "long double")) 
              << ", ur=" << (precr == 1 ? "single" : "long double") 
              << ", ug=" << (precw == 1 ? "single" : (precw == 2 ? "double" : "long double")) << "):\n";

    // LU factorization in Tf
    Tf* A_tf = allocate_matrix<Tf>(n, n);
    for (int i = 0; i < n * n; ++i) A_tf[i] = static_cast<Tf>(A[i]);
    Tf* L = allocate_matrix<Tf>(n, n);
    Tf* U = allocate_matrix<Tf>(n, n);
    int* P = new int[n];
    lu_factorization(A_tf, L, U, P, n);


    // Initial solution x_0 = U^-1 L^-1 b in Tf
    Tf* x = allocate_matrix<Tf>(n, 1);
    if (L != nullptr && U != nullptr && P != nullptr) {
        Tf* b_tf = new Tf[n];
        for (int i = 0; i < n; ++i) b_tf[P[i]] = static_cast<Tf>(b[i]);
        Tf* y = new Tf[n];
        forward_substitution(L, b_tf, y, n);
        backward_substitution(U, y, x, n);
        delete[] b_tf;
        delete[] y;
    } else {
        for (int i = 0; i < n; ++i) x[i] = Tf(0);
    }

    // Check for Inf in initial solution
    bool has_inf = false;
    for (int i = 0; i < n; ++i) {
        if (std::isinf(static_cast<double>(x[i]))) {
            has_inf = true;
            break;
        }
    }
    if (has_inf) {
        std::cout << "**** Warning: x0 contains Inf. Using 0 vector as initial solution.\n";
        for (int i = 0; i < n; ++i) x[i] = Tf(0);
    }

    // Store solution in working precision Tu
    Tu* x_u = allocate_matrix<Tu>(n, 1);
    Tu* x_old = allocate_matrix<Tu>(n, 1);
    for (int i = 0; i < n; ++i) x_u[i] = static_cast<Tu>(x[i]);

    // Arrays for convergence metrics and GMRES tracking
    double* ferr = new double[max_iter + 1];
    double* nbe = new double[max_iter + 1];
    double* cbe = new double[max_iter + 1];
    int* gmresits = new int[max_iter];
    double* gmreserr = new double[max_iter];

    Tr* r = allocate_matrix<Tr>(n, 1);
    Tg* d = allocate_matrix<Tg>(n, 1);

    int iter = 0;
    bool cged = false;
    double convergence_threshold = u * kappa;
    while (!cged) {
        // Compute convergence metrics
        ferr[iter] = 0.0;
        for (int i = 0; i < n; ++i) {
            double err = std::abs(static_cast<double>(x_u[i]) - x_true[i]);
            if (err > ferr[iter]) ferr[iter] = err;
        }
        double x_true_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(x_true[i]) > x_true_norm) x_true_norm = std::abs(x_true[i]);
        }
        ferr[iter] /= x_true_norm;

        // Compute residual r = b - Ax in Tr (high precision)
        for (int i = 0; i < n; ++i) {
            Tr sum = static_cast<Tr>(b[i]);
            for (int j = 0; j < n; ++j) {
                sum -= static_cast<Tr>(A[i * n + j]) * static_cast<Tr>(x_u[j]);
            }
            r[i] = sum;
        }

        // Normwise backward error
        double r_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(static_cast<double>(r[i])) > r_norm) r_norm = std::abs(static_cast<double>(r[i]));
        }
        double A_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < n; ++j) {
                row_sum += std::abs(A[i * n + j]);
            }
            if (row_sum > A_norm) A_norm = row_sum;
        }
        double x_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(static_cast<double>(x_u[i])) > x_norm) x_norm = std::abs(static_cast<double>(x_u[i]));
        }
        double b_norm = 0.0;
        for (int i = 0; i < n; ++i) {
            if (std::abs(b[i]) > b_norm) b_norm = std::abs(b[i]);
        }
        nbe[iter] = r_norm / (A_norm * x_norm + b_norm);

        // Componentwise backward error
        double* temp = new double[n];
        for (int i = 0; i < n; ++i) {
            double axb = 0.0;
            for (int j = 0; j < n; ++j) {
                axb += std::abs(A[i * n + j]) * std::abs(static_cast<double>(x_u[j]));
            }
            axb += std::abs(b[i]);
            temp[i] = axb > 0 ? std::abs(static_cast<double>(r[i])) / axb : 0.0;
        }
        cbe[iter] = 0.0;
        for (int i = 0; i < n; ++i) {
            if (temp[i] > cbe[iter]) cbe[iter] = temp[i];
        }
        delete[] temp;

        std::cout << "Iteration " << iter + 1 << ": ferr=" << ferr[iter] << ", nbe=" << nbe[iter] << ", cbe=" << cbe[iter] << "\n";

        if (iter >= max_iter) break;

        // Check convergence
        if (std::max(std::max(ferr[iter], nbe[iter]), cbe[iter]) <= convergence_threshold) {
            cged = true;
            break;
        }

        // Debug: Print residual
        std::cout << "Residual at iteration " << iter + 1 << ": ";
        for (int i = 0; i < n; ++i) {
            std::cout << static_cast<double>(r[i]) << " ";
        }
        std::cout << "\n";

        // GMRES solve for correction d
        int gmres_iter = 0;
        double gmres_err = 0.0;
        gmres<Tf, Tr, Tg>(A_tf, d, r, L, U, P, n, n, gtol, &gmres_iter, &gmres_err);

        // Debug: Print correction
        std::cout << "Correction at iteration " << iter + 1 << ": ";
        for (int i = 0; i < n; ++i) {
            std::cout << static_cast<double>(d[i]) << " ";
        }
        std::cout << "\n";

        // Check GMRES residual norm
        if (gmres_err < 1e-12) {
            std::cout << "Warning: GMRES residual norm too small (" << gmres_err << "), may indicate numerical issues\n";
        }

        // Update x = x + d
        for (int i = 0; i < n; ++i) x_old[i] = x_u[i];
        for (int i = 0; i < n; ++i) {
            x_u[i] = x_u[i] + static_cast<Tu>(d[i]);
        }

        // Check dx for Inf or NaN
        double dx_norm = 0.0;
        double x_norm_new = 0.0;
        for (int i = 0; i < n; ++i) {
            double dx = std::abs(static_cast<double>(x_u[i]) - static_cast<double>(x_old[i]));
            if (dx > dx_norm) dx_norm = dx;
            if (std::abs(static_cast<double>(x_u[i])) > x_norm_new) x_norm_new = std::abs(static_cast<double>(x_u[i]));
        }
        double dx = x_norm_new > 0 ? dx_norm / x_norm_new : dx_norm;
        if (std::isinf(dx) || std::isnan(dx)) {
            std::cout << "**** Warning: dx contains Inf or NaN. Stopping.\n";
            break;
        }

        // Stop if ferr stagnates or grows slightly
        if (iter > 1 && ferr[iter] > ferr[iter-1] && (ferr[iter] < ferr[iter-1] * 1.1)) {
            std::cout << "**** Warning: Forward error stagnating. Stopping.\n";
            break;
        }

        iter++;
    }

    // Print final results
    std::cout << "Final solution: ";
    for (int i = 0; i < n; ++i) {
        std::cout << static_cast<double>(x_u[i]) << " ";
    }
    std::cout << "\nTrue solution: ";
    for (int i = 0; i < n; ++i) {
        std::cout << x_true[i] << " ";
    }
    std::cout << "\nGMRES iterations: ";
    for (int i = 0; i < iter; ++i) {
        std::cout << gmresits[i] << " ";
    }
    std::cout << "\nGMRES final residual norms: ";
    for (int i = 0; i < iter; ++i) {
        std::cout << gmreserr[i] << " ";
    }
    std::cout << "\n";

    deallocate_matrix(A_tf);
    if (L != nullptr) deallocate_matrix(L);
    if (U != nullptr) deallocate_matrix(U);
    delete[] P;
    deallocate_matrix(x);
    deallocate_matrix(r);
    deallocate_matrix(d);
    deallocate_matrix(x_u);
    deallocate_matrix(x_old);
    delete[] ferr;
    delete[] nbe;
    delete[] cbe;
    delete[] gmresits;
    delete[] gmreserr;
}

// Test function
void test_gmres_ir(double kappa, int precf, int precw, int precr, int max_iter, double gtol) {
    int n = 4;
    
    double* x_true = new double[n];
    for (int i = 0; i < n; ++i) {
        x_true[i] = 1.0;
    }
    
    double* A = gallery_randsvd(n, kappa);
    std::cout << "\nMatrix A (kappa=" << kappa << "):\n";
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << std::fixed << std::setprecision(6) << get_element(A, n, n, i, j) << " ";
        }
        std::cout << "\n";
    }
    
    double* b = new double[n]();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b[i] += get_element(A, n, n, i, j) * x_true[j];
        }
    }
    
    // Test precision combinations
    if (precf == 1 && precw == 1 && precr == 1) {
        gmres_ir<float, float, float, float>(A, b, x_true, n, kappa, precf, precw, precr, max_iter, gtol);
    } else if (precf == 1 && precw == 1 && precr == 2) {
        gmres_ir<float, long double, float, float>(A, b, x_true, n, kappa, precf, precw, precr, max_iter, gtol);
    } else if (precf == 2 && precw == 2 && precr == 2) {
        gmres_ir<double, long double, double, double>(A, b, x_true, n, kappa, precf, precw, precr, max_iter, gtol);
    } else if (precf == 2 && precw == 3 && precr == 2) {
        gmres_ir<double, long double, long double, long double>(A, b, x_true, n, kappa, precf, precw, precr, max_iter, gtol);
    } else {
        std::cout << "Unsupported precision combination\n";
    }
    
    deallocate_matrix(A);
    delete[] b;
    delete[] x_true;
}

int main() {
    double kappas[] = {10.0, 100.0, 10000.0};
    int max_iter = 100;
    double gtol = 1e-12;
    
    for (double kappa : kappas) {
        std::cout << "\nTesting with kappa = " << kappa << "\n";
        test_gmres_ir(kappa, 1, 1, 1, max_iter, gtol); // Single, single, single
        test_gmres_ir(kappa, 1, 1, 2, max_iter, gtol); // Single, single, long double
        test_gmres_ir(kappa, 2, 2, 2, max_iter, gtol); // Double, double, long double
        test_gmres_ir(kappa, 2, 3, 2, max_iter, gtol); // Double, long double, long double
    }
    return 0;
}