#include <half.hpp>
#include <floatx.hpp>
#include <iostream>
#include <cmath>




float true_function(float x) {
    // True function for generating data points and exact values
    return sin(x);
}


void solve_tridiagonal(double* a, double* b, double* c, double* d, double* x, int n) {
    // Solve tridiagonal system using Thomas algorithm
    // Forward elimination
    float* c_prime = new float[n];
    float* d_prime = new float[n];
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];
    for (int i = 1; i < n; ++i) {
        float m = b[i] - a[i] * c_prime[i - 1];
        c_prime[i] = c[i] / m;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / m;
    }
    // Back substitution
    x[n - 1] = d_prime[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
    delete[] c_prime;
    delete[] d_prime;
}


void compute_spline_coefficients(double* x, double* y, int n,
                                 double** a, double** b, double** c, double** d) {
    // Compute cubic spline coefficients
    *a = new double[n - 1];
    *b = new double[n - 1];
    *c = new double[n - 1];
    *d = new double[n - 1];
    
    // Compute h_i = x_{i+1} - x_i
    float* h = new float[n - 1];
    for (int i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
    }

    // Set up tridiagonal system for second derivatives k_i
    double* k = new double[n];
    k[0] = 0.0; // Natural spline: k_0 = 0
    k[n - 1] = 0.0; // k_{n-1} = 0
    if (n > 2) {
        double* diag = new double[n - 2]; // Main diagonal
        double* subdiag = new double[n - 2]; // Subdiagonal
        double* superdiag = new double[n - 2]; // Superdiagonal
        double* rhs = new double[n - 2]; // Right-hand side
        for (int i = 0; i < n - 2; ++i) {
            diag[i] = 2.0 * (h[i] + h[i + 1]);
            subdiag[i] = h[i];
            superdiag[i] = h[i + 1];
            rhs[i] = 6.0 * ((y[i + 2] - y[i + 1]) / h[i + 1] - (y[i + 1] - y[i]) / h[i]);
        }
        // Solve for k_1, ..., k_{n-2}
        solve_tridiagonal(subdiag, diag, superdiag, rhs, k + 1, n - 2);
        delete[] diag;
        delete[] subdiag;
        delete[] superdiag;
        delete[] rhs;
    }

    // Compute coefficients
    for (int i = 0; i < n - 1; ++i) {
        (*a)[i] = y[i];
        (*c)[i] = k[i] / 2.0;
        (*d)[i] = (k[i + 1] - k[i]) / (6.0 * h[i]);
        (*b)[i] = (y[i + 1] - y[i]) / h[i] + h[i] * (k[i] + 2.0 * k[i + 1]) / 6.0;
    }

    delete[] h;
    delete[] k;
}

// Evaluate spline at x_eval
float evaluate_spline(double x_eval, double* x, double* a, double* b, double* c, double* d, int n) {
    // Find interval [x_i, x_{i+1}] containing x_eval
    int i = 0;
    while (i < n - 1 && x[i + 1] < x_eval) {
        ++i;
    }
    if (i >= n - 1) i = n - 2; // Handle boundary

    float dx = x_eval - x[i];
    return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
}


int main() {
    
    int n = 20; // Test different numbers of points

    int num_eval = 100; // Evaluation points
    flx::floatx<5, 2> x_start = 0.0;
    double x_end = 4.5;

    std::cout << "n, Max Abs Error, Mean Abs Error, RMSE, Max Rel Error\n";
    double* x = new double[n]; // Generate data points
    double* y = new double[n];
    float h = (x_end - x_start) / (n - 1);
    for (int i = 0; i < n; ++i) {
        x[i] = x_start + i * h;
        y[i] = true_function(x[i]);
    }

    double* a = nullptr; // Compute spline coefficients
    double* b = nullptr;
    double* c = nullptr;
    double* d = nullptr;
    compute_spline_coefficients(x, y, n, &a, &b, &c, &d);

    double* x_eval = new double[num_eval]; // Evaluate spline on fine grid
    double* y_interp = new double[num_eval];
    double h_eval = (x_end - x_start) / (num_eval - 1);
    for (int i = 0; i < num_eval; ++i) {
        x_eval[i] = x_start + i * h_eval;
        y_interp[i] = evaluate_spline(x_eval[i], x, a, b, c, d, n);
    }

    PROMISE_CHECK_ARRAY(y_interp, num_eval);
    delete[] x;
    delete[] y;
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
    delete[] x_eval;
    delete[] y_interp;

    return 0;
}