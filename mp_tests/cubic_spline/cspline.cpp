#include <iostream>
#include <cmath>




__PROMISE__ true_function(__PROMISE__ x) {
    // True function for generating data points and exact values
    return sin(x);
}


void solve_tridiagonal(__PROMISE__* a, __PROMISE__* b, __PROMISE__* c, __PROMISE__* d, __PROMISE__* x, int n) {
    // Solve tridiagonal system using Thomas algorithm
    // Forward elimination
    __PROMISE__* c_prime = new __PROMISE__[n];
    __PROMISE__* d_prime = new __PROMISE__[n];
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];
    for (int i = 1; i < n; ++i) {
        __PROMISE__ m = b[i] - a[i] * c_prime[i - 1];
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


void compute_spline_coefficients(__PROMISE__* x, __PROMISE__* y, int n,
                                 __PROMISE__** a, __PROMISE__** b, __PROMISE__** c, __PROMISE__** d) {
    // Compute cubic spline coefficients
    *a = new __PROMISE__[n - 1];
    *b = new __PROMISE__[n - 1];
    *c = new __PROMISE__[n - 1];
    *d = new __PROMISE__[n - 1];
    
    // Compute h_i = x_{i+1} - x_i
    __PROMISE__* h = new __PROMISE__[n - 1];
    for (int i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
    }

    // Set up tridiagonal system for second derivatives k_i
    __PROMISE__* k = new __PROMISE__[n];
    k[0] = 0.0; // Natural spline: k_0 = 0
    k[n - 1] = 0.0; // k_{n-1} = 0
    if (n > 2) {
        __PROMISE__* diag = new __PROMISE__[n - 2]; // Main diagonal
        __PROMISE__* subdiag = new __PROMISE__[n - 2]; // Subdiagonal
        __PROMISE__* superdiag = new __PROMISE__[n - 2]; // Superdiagonal
        __PROMISE__* rhs = new __PROMISE__[n - 2]; // Right-hand side
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
__PROMISE__ evaluate_spline(__PROMISE__ x_eval, __PROMISE__* x, __PROMISE__* a, __PROMISE__* b, __PROMISE__* c, __PROMISE__* d, int n) {
    // Find interval [x_i, x_{i+1}] containing x_eval
    int i = 0;
    while (i < n - 1 && x[i + 1] < x_eval) {
        ++i;
    }
    if (i >= n - 1) i = n - 2; // Handle boundary

    __PROMISE__ dx = x_eval - x[i];
    return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
}


int main() {
    
    int n = 20; // Test different numbers of points

    int num_eval = 100; // Evaluation points
    __PROMISE__ x_start = 0.0;
    __PROMISE__ x_end = 4.5;

    std::cout << "n, Max Abs Error, Mean Abs Error, RMSE, Max Rel Error\n";
    __PROMISE__* x = new __PROMISE__[n]; // Generate data points
    __PROMISE__* y = new __PROMISE__[n];
    __PROMISE__ h = (x_end - x_start) / (n - 1);
    for (int i = 0; i < n; ++i) {
        x[i] = x_start + i * h;
        y[i] = true_function(x[i]);
    }

    __PROMISE__* a = nullptr; // Compute spline coefficients
    __PROMISE__* b = nullptr;
    __PROMISE__* c = nullptr;
    __PROMISE__* d = nullptr;
    compute_spline_coefficients(x, y, n, &a, &b, &c, &d);

    __PROMISE__* x_eval = new __PROMISE__[num_eval]; // Evaluate spline on fine grid
    __PROMISE__* y_interp = new __PROMISE__[num_eval];
    __PROMISE__ h_eval = (x_end - x_start) / (num_eval - 1);
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