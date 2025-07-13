#include <iostream>
#include <cmath>

void copy(half_float::half* dest, half_float::half* src, int n) {
    for (int i = 0; i < n; ++i) {
        dest[i] = src[i];
    }
}

void axpy(half_float::half a, half_float::half* x, half_float::half* y, int n) {
    // y = a * x + y
    for (int i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
}

void scale(half_float::half a, half_float::half* x, int n) {
    // x = a * x
    for (int i = 0; i < n; ++i) {
        x[i] *= a;
    }
}

void add(half_float::half* x, half_float::half* y, half_float::half* result, int n) {
    // result = x + y
    for (int i = 0; i < n; ++i) {
        result[i] = x[i] + y[i];
    }
}


double true_function(double x) {
    // True function for generating data points and exact values
    return sin(x);
}


float lagrange_interpolate(double x_eval, double* x, double* y, int n) {
    // Lagrange interpolation at x_eval
    double result = 0.0;
    for (int i = 0; i < n; ++i) {
        double li = 1.0; // Lagrange basis polynomial l_i(x)
        for (int j = 0; j < n; ++j) {
            if (j != i) {
                li *= (x_eval - x[j]) / (x[i] - x[j]);
            }
        }
        result += y[i] * li;
    }
    return result;
}


void compute_metrics(double* x_eval, double* y_interp, int num_eval, double* max_abs_error,
                     double* mean_abs_error, double* rmse, double* max_rel_error) {
                     // Compute evaluation metrics
    float sum_abs_error = 0.0;
    float sum_sq_error = 0.0;
    *max_abs_error = 0.0;
    *max_rel_error = 0.0;

    for (int i = 0; i < num_eval; ++i) {
        float y_exact = true_function(x_eval[i]);
        float error = fabs(y_interp[i] - y_exact);
        sum_abs_error += error;
        sum_sq_error += error * error;
        if (error > *max_abs_error) {
            *max_abs_error = error;
        }
        if (fabs(y_exact) > 1e-10) {
            float rel_error = error / fabs(y_exact);
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
    float x_start = 0.0;
    double x_end = 4.5;

    std::cout << "n, Max Abs Error, Mean Abs Error, RMSE, Max Rel Error\n";

    // Generate data points
    double* x = new double[n];
    double* y = new double[n];
    float h = (x_end - x_start) / (n - 1);
    for (int i = 0; i < n; ++i) {
        x[i] = x_start + i * h;
        y[i] = true_function(x[i]);
    }

    double* x_eval = new double[num_eval];  // Evaluate Lagrange interpolation on fine grid
    double* y_interp = new double[num_eval];
    double h_eval = (x_end - x_start) / (num_eval - 1);

    for (int i = 0; i < num_eval; ++i) {
        x_eval[i] = x_start + i * h_eval;
        y_interp[i] = lagrange_interpolate(x_eval[i], x, y, n);
    }

    double max_abs_error, mean_abs_error, rmse, max_rel_error;
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