#include <iostream>
#include <cmath>

void copy(double* dest, double* src, int n) {
    for (int i = 0; i < n; ++i) {
        dest[i] = src[i];
    }
}

void axpy(double a, double* x, double* y, int n) {
    // y = a * x + y
    for (int i = 0; i < n; ++i) {
        y[i] += a * x[i];
    }
}

void scale(flx::floatx<4, 3> a, flx::floatx<4, 3>* x, int n) {
    // x = a * x
    for (int i = 0; i < n; ++i) {
        x[i] *= a;
    }
}

void add(flx::floatx<4, 3>* x, flx::floatx<4, 3>* y, flx::floatx<4, 3>* result, int n) {
    // result = x + y
    for (int i = 0; i < n; ++i) {
        result[i] = x[i] + y[i];
    }
}


void ode_function(float t, double* y, double* dydt, int n) {
    // ODE: dy_i/dt = y_{i-1} - 2*y_i + y_{i+1} (tridiagonal system)
    dydt[0] = -2.0 * y[0] + y[1];
    for (int i = 1; i < n - 1; ++i) {
        dydt[i] = y[i - 1] - 2.0 * y[i] + y[i + 1];
    }
    dydt[n - 1] = y[n - 2] - 2.0 * y[n - 1];
}


void analytical_solution(float t, double* y_exact, int n) {
    // Analytical solution: for n=2, exact; for n>2, use RK4 with small h
    if (n == 2) {
        y_exact[0] = 0.5 * exp(-t) + 0.5 * exp(-3.0 * t);
        y_exact[1] = 0.5 * exp(-t) - 0.5 * exp(-3.0 * t);
    } else {
        // Use RK4 with small h (h_ref = 0.0001) as reference
        float t0 = 0.0;
        float h_ref = 0.0001;
        int steps = static_cast<int>(t / h_ref) + 1;
        double* y = new double[n];
        double* y_new = new double[n];
        y[0] = 1.0;
        for (int i = 1; i < n; ++i) y[i] = 0.0;
        float current_t = t0;
        for (int i = 0; i < steps && current_t < t; ++i) {
            // RK4 step for reference solution
            double* k1 = new double[n];
            double* k2 = new double[n];
            double* k3 = new double[n];
            double* k4 = new double[n];
            double* temp = new double[n];

            ode_function(current_t, y, k1, n);
            copy(temp, y, n);
            axpy(h_ref / 2.0, k1, temp, n);
            ode_function(current_t + h_ref / 2.0, temp, k2, n);
            copy(temp, y, n);
            axpy(h_ref / 2.0, k2, temp, n);
            ode_function(current_t + h_ref / 2.0, temp, k3, n);
            copy(temp, y, n);
            axpy(h_ref, k3, temp, n);
            ode_function(current_t + h_ref, temp, k4, n);
            copy(y_new, y, n);
            axpy(h_ref / 6.0, k1, y_new, n);
            axpy(h_ref * 2.0 / 6.0, k2, y_new, n);
            axpy(h_ref * 2.0 / 6.0, k3, y_new, n);
            axpy(h_ref / 6.0, k4, y_new, n);

            delete[] k1;
            delete[] k2;
            delete[] k3;
            delete[] k4;
            delete[] temp;

            copy(y, y_new, n);
            current_t += h_ref;
        }
        copy(y_exact, y, n);
        delete[] y;
        delete[] y_new;
    }
}

void trapezoidal_step(float t, double h, double* y, int n, double* y_new) {

    // Trapezoidal Rule step with fixed-point iteration
    int max_iterations = 5; // Fixed number of iterations
    double* f_n = new double[n];
    double* f_np1 = new double[n];
    double* y_guess = new double[n];
    flx::floatx<4, 3>* temp = new flx::floatx<4, 3>[n];

    // Compute f(t_n, y_n)
    ode_function(t, y, f_n, n);

    // Initial guess: y_n
    copy(y_guess, y, n);

    // Fixed-point iteration: y_{n+1} = y_n + (h/2) * (f(t_n, y_n) + f(t_{n+1}, y_{n+1}))
    for (int iter = 0; iter < max_iterations; ++iter) {
        ode_function(t + h, y_guess, f_np1, n);
        copy(y_new, y, n); // y_new = y_n
        axpy(h / 2.0, f_n, y_new, n); // y_new += (h/2) * f_n
        axpy(h / 2.0, f_np1, y_new, n); // y_new += (h/2) * f_np1
        copy(y_guess, y_new, n); // Update guess for next iteration
    }

    delete[] f_n;
    delete[] f_np1;
    delete[] y_guess;
    delete[] temp;
}


void trapezoidal_solve(float t0, float tf, double h, double* y0, int n,
                       double* results, int* num_steps) {
    if (*num_steps <= 0) return;

    double* y = new double[n];
    double* y_new = new double[n];

    copy(y, y0, n);
    copy(results, y0, n);
    float t = t0;

    for (int i = 1; i < *num_steps; ++i) {
        trapezoidal_step(t, h, y, n, y_new);
        copy(y, y_new, n);
        copy(results + i * n, y, n);
        t += h;
    }

    delete[] y;
    delete[] y_new;
}


void compute_metrics(float t0, float h, double* results, int n, int num_steps,
                     double* max_abs_error, double* mean_abs_error, double* rmse,
                     double* max_rel_error) {
    double* y_exact = new double[n];
    float sum_abs_error = 0.0;
    float sum_sq_error = 0.0;
    *max_abs_error = 0.0;
    *max_rel_error = 0.0;
    int total_points = num_steps * n;

    for (int i = 0; i < num_steps; ++i) {
        float t = t0 + i * h;
        analytical_solution(t, y_exact, n);
        for (int j = 0; j < n; ++j) {
            float error = fabs(results[i * n + j] - y_exact[j]);
            sum_abs_error += error;
            sum_sq_error += error * error;
            if (error > *max_abs_error) {
                *max_abs_error = error;
            }
            if (fabs(y_exact[j]) > 1e-10) {
                float rel_error = error / fabs(y_exact[j]);
                if (rel_error > *max_rel_error) {
                    *max_rel_error = rel_error;
                }
            }
        }
    }

    *mean_abs_error = sum_abs_error / total_points;
    *rmse = sqrt(sum_sq_error / total_points);
    delete[] y_exact;
}

int main() {
    int n = 10;
    float t0 = 0.0;
    float tf = 1.0;
    double* y0 = new double[n];
    y0[0] = 1.0;
    for (int i = 1; i < n; ++i) y0[i] = 0.0;


    double h = 0.001;
    std::cout << "h, Max Abs Error, Mean Abs Error, RMSE, Max Rel Error\n";

    int num_steps = static_cast<int>((tf - t0) / h) + 1;
    double* results = new double[num_steps * n];

    trapezoidal_solve(t0, tf, h, y0, n, results, &num_steps);

    double max_abs_error, mean_abs_error, rmse, max_rel_error;
    compute_metrics(t0, h, results, n, num_steps,
                    &max_abs_error, &mean_abs_error, &rmse, &max_rel_error);

    std::cout << h << ", " << max_abs_error << ", " << mean_abs_error << ", "
                << rmse << ", " << max_rel_error << "\n";

    PROMISE_CHECK_ARRAY(results, num_steps * n);
    delete[] results;

    delete[] y0;
    return 0;
}
    