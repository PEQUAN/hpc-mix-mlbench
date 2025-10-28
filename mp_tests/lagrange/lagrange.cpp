#include <iostream>
#include <cmath>

void copy(__PROMISE__* dest, __PROMISE__* src, int n) {
    for (int i = 0; i < n; ++i) {
        dest[i] = src[i];
    }
}

void axpy(__PROMISE__ a, __PROMISE__* x, __PROMISE__* y, int n) {
    // y = a * x + y
    for (int i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
}

void scale(__PROMISE__ a, __PROMISE__* x, int n) {
    // x = a * x
    for (int i = 0; i < n; ++i) {
        x[i] *= a;
    }
}

void add(__PROMISE__* x, __PROMISE__* y, __PROMISE__* result, int n) {
    // result = x + y
    for (int i = 0; i < n; ++i) {
        result[i] = x[i] + y[i];
    }
}


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


void compute_metrics(__PROMISE__* x_eval, __PROMISE__* y_interp, int num_eval, __PROMISE__* max_abs_error,
                     __PROMISE__* mean_abs_error, __PROMISE__* rmse, __PROMISE__* max_rel_error) {
                     // Compute evaluation metrics
    __PROMISE__ sum_abs_error = 0.0;
    __PROMISE__ sum_sq_error = 0.0;
    *max_abs_error = 0.0;
    *max_rel_error = 0.0;

    for (int i = 0; i < num_eval; ++i) {
        __PROMISE__ y_exact = true_function(x_eval[i]);
        __PROMISE__ error = fabs(y_interp[i] - y_exact);
        sum_abs_error += error;
        sum_sq_error += error * error;
        if (error > *max_abs_error) {
            *max_abs_error = error;
        }
        if (fabs(y_exact) > 1e-10) {
            __PROMISE__ rel_error = error / fabs(y_exact);
            if (rel_error > *max_rel_error) {
                *max_rel_error = rel_error;
            }
        }
    }

    *mean_abs_error = sum_abs_error / num_eval;
    *rmse = sqrt(sum_sq_error / num_eval);
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

    __PROMISE__ max_abs_error, mean_abs_error, rmse, max_rel_error;
    compute_metrics(x_eval, y_interp, num_eval, &max_abs_error, &mean_abs_error, &rmse, &max_rel_error);

    std::cout << n << ", " << max_abs_error << ", " << mean_abs_error << ", "
                << rmse << ", " << max_rel_error << "\n";

    PROMISE_CHECK_ARRAY(y_interp, num_eval);
    delete[] x;
    delete[] y;
    delete[] x_eval;
    delete[] y_interp;

    return 0;
}