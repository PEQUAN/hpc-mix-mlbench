#include <iostream>
#include <cmath>

__PROMISE__ true_function(__PROMISE__ x) {
    // True function for generating data points and exact values
    return sin(x);
}


__PROMISE__ lagrange_interpolate(__PROMISE__ x_eval, __PROMISE__* x, __PROMISE__* y, int n) {
    // Lagrange interpolation at x_eval
    __PROMISE__ result = 0.0;
    for (int i = 0; i < n; ++i) {
        __PROMISE__ li = 1.0; // Lagrange basis polynomial l_i(x)
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                li *= (x_eval - x[j]) / (x[i] - x[j]);
            }
        }
        result += y[i] * li;
    }
    return result;
}


int main() {
    int n= 20;
    int num_eval = 100; 
    __PROMISE__ x_start = 0.0;
    __PROMISE__ x_end = 4.5;

    std::cout << "n, Max Abs Error, Mean Abs Error, RMSE, Max Rel Error\n";

    // Generate data points
    __PROMISE__* x = new __PROMISE__[n];
    __PROMISE__* y = new __PROMISE__[n];
    __PROMISE__ h = (x_end - x_start) / (n - 1);
    for (int i = 0; i < n; ++i) {
        x[i] = x_start + i * h;
        y[i] = true_function(x[i]);
    }

    __PROMISE__* x_eval = new __PROMISE__[num_eval];  // Evaluate Lagrange interpolation on fine grid
    __PROMISE__* y_interp = new __PROMISE__[num_eval];
    __PROMISE__ h_eval = (x_end - x_start) / (num_eval - 1);

    for (int i = 0; i < num_eval; ++i) {
        x_eval[i] = x_start + i * h_eval;
        y_interp[i] = lagrange_interpolate(x_eval[i], x, y, n);
    }


    PROMISE_CHECK_ARRAY(y_interp, num_eval);
    delete[] x;
    delete[] y;
    delete[] x_eval;
    delete[] y_interp;

    return 0;
}