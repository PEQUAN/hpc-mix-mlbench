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

void scale(double a, double* x, int n) {
    // x = a * x
    for (int i = 0; i < n; ++i) {
        x[i] *= a;
    }
}

void add(double* x, double* y, double* result, int n) {
    // result = x + y
    for (int i = 0; i < n; ++i) {
        result[i] = x[i] + y[i];
    }
}



void ode_function(double t, double* y, double* dydt, int n) {
    // ODE: dy_i/dt = y_{i-1} - 2*y_i + y_{i+1} (tridiagonal system)
    dydt[0] = -2.0 * y[0] + y[1];
    for (int i = 1; i < n - 1; ++i) {
        dydt[i] = y[i - 1] - 2.0 * y[i] + y[i + 1];
    }
    dydt[n - 1] = y[n - 2] - 2.0 * y[n - 1];
}

void rk4_step(double t, double h, double* y, int n, double* y_new) {
    // RK4 step
    double* k1 = new double[n];
    double* k2 = new double[n];
    double* k3 = new double[n];
    double* k4 = new double[n];
    double* temp = new double[n];

    ode_function(t, y, k1, n);
    copy(temp, y, n);
    axpy(h / 2.0, k1, temp, n);
    ode_function(t + h / 2.0, temp, k2, n);
    copy(temp, y, n);
    axpy(h / 2.0, k2, temp, n);
    ode_function(t + h / 2.0, temp, k3, n);
    copy(temp, y, n);
    axpy(h, k3, temp, n);
    ode_function(t + h, temp, k4, n);
    copy(y_new, y, n);
    axpy(h / 6.0, k1, y_new, n);
    axpy(h * 2.0 / 6.0, k2, y_new, n);
    axpy(h * 2.0 / 6.0, k3, y_new, n);
    axpy(h / 6.0, k4, y_new, n);

    delete[] k1;
    delete[] k2;
    delete[] k3;
    delete[] k4;
    delete[] temp;
}



void analytical_solution(double t, double* y_exact, int n) {
    // Analytical solution: for n=2, exact; for n>2, use RK4 with small h
    if (n == 2) {
        y_exact[0] = 0.5 * exp(-t) + 0.5 * exp(-3.0 * t);
        y_exact[1] = 0.5 * exp(-t) - 0.5 * exp(-3.0 * t);
    } else {
        // Use RK4 with small h (h_ref = 0.0001) as reference
        double t0 = 0.0;
        double h_ref = 0.0001;
        int steps = static_cast<int>(t / h_ref) + 1;
        double* y = new double[n];
        double* y_new = new double[n];
        y[0] = 1.0;
        for (int i = 1; i < n; ++i) y[i] = 0.0;
        double current_t = t0;
        for (int i = 0; i < steps && current_t < t; ++i) {
            rk4_step(current_t, h_ref, y, n, y_new);
            copy(y, y_new, n);
            current_t += h_ref;
        }
        copy(y_exact, y, n);
        delete[] y;
        delete[] y_new;
    }
}

void rk4_solve(double t0, double tf, double h, double* y0, int n,
               double* results, int* num_steps) {

    // RK4 solver
    *num_steps = static_cast<int>((tf - t0) / h) + 1;
    if (*num_steps <= 0) return;

    results = new double[*num_steps * n];
    double* y = new double[n];
    double* y_new = new double[n];

    copy(y, y0, n);
    copy(results, y0, n);
    double t = t0;

    for (int i = 1; i < *num_steps; ++i) {
        rk4_step(t, h, y, n, y_new);
        copy(y, y_new, n);
        copy(results + i * n, y, n);
        t += h;
    }

    delete[] y;
    delete[] y_new;
}


void compute_metrics(double t0, double h, double* results, int n, int num_steps,
                     double* max_abs_error, double* mean_abs_error, double* rmse,
                     double* max_rel_error) {
    double* y_exact = new double[n];
    double sum_abs_error = 0.0;
    double sum_sq_error = 0.0;
    *max_abs_error = 0.0;
    *max_rel_error = 0.0;
    int total_points = num_steps * n;

    for (int i = 0; i < num_steps; ++i) {
        double t = t0 + i * h;
        analytical_solution(t, y_exact, n);
        for (int j = 0; j < n; ++j) {
            double error = fabs(results[i * n + j] - y_exact[j]);
            sum_abs_error += error;
            sum_sq_error += error * error;
            if (error > *max_abs_error) {
                *max_abs_error = error;
            }
            // Skip relative error if y_exact is too small to avoid inf
            if (fabs(y_exact[j]) > 1e-10) {
                double rel_error = error / fabs(y_exact[j]);
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
    // Problem setup
    int n = 10; // Changed to n=10
    double t0 = 0.0;
    double tf = 1.0;
    double* y0 = new double[n];
    y0[0] = 1.0;
    for (int i = 1; i < n; ++i) y0[i] = 0.0;

    double h = 0.001; // Test different step sizes

    std::cout << "h, Max Abs Error, Mean Abs Error, RMSE, Max Rel Error\n";

    
    int num_steps = static_cast<int>((tf - t0) / h) + 1;;
    double* results = new double[num_steps * n];
    rk4_solve(t0, tf, h, y0, n, results, &num_steps);

    
    double max_abs_error, mean_abs_error, rmse, max_rel_error;
    compute_metrics(t0, h, results, n, num_steps,
                    &max_abs_error, &mean_abs_error, &rmse, &max_rel_error);


    std::cout << h << ", " << max_abs_error << ", " << mean_abs_error << ", "
                << rmse << ", " << max_rel_error << "\n";

    
    delete[] results;
  
    delete[] y0;
    return 0;
}