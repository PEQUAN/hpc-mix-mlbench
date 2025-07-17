#include <iostream>
#include <cmath>

void copy(float* dest, float* src, int n) {
    for (int i = 0; i < n; ++i) {
        dest[i] = src[i];
    }
}

void axpy(float a, float* x, float* y, int n) {
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

void ode_function(float t, float* y, float* dydt, int n) {
    // ODE: dy_i/dt = y_{i-1} - 2*y_i + y_{i+1} (tridiagonal system)
    dydt[0] = -2.0 * y[0] + y[1];
    for (int i = 1; i < n - 1; ++i) {
        dydt[i] = y[i - 1] - 2.0 * y[i] + y[i + 1];
    }
    dydt[n - 1] = y[n - 2] - 2.0 * y[n - 1];
}

void analytical_solution(float t, float* y_exact, int n) {
    // Analytical solution: for n=2, exact; for n>2, use RK4 with small h
    if (n == 2) {
        y_exact[0] = 0.5 * exp(-t) + 0.5 * exp(-3.0 * t);
        y_exact[1] = 0.5 * exp(-t) - 0.5 * exp(-3.0 * t);
    } else {
        // Use RK4 with small h (h_ref = 0.0001) as reference
        float t0 = 0.0;
        float h_ref = 0.0001;
        int steps = static_cast<int>(t / h_ref) + 1;
        float* y = new float[n];
        float* y_new = new float[n];
        y[0] = 1.0;
        for (int i = 1; i < n; ++i) y[i] = 0.0;
        float current_t = t0;
        for (int i = 0; i < steps && current_t < t; ++i) {
            // RK4 step for reference solution
            float* k1 = new float[n];
            float* k2 = new float[n];
            float* k3 = new float[n];
            float* k4 = new float[n];
            float* temp = new float[n];

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

// Simpson's Rule step 
void simpsons_step(float t, float h, float* y_n, int n, float* y_np2) {
    int max_iterations = 5;
    float* f_n = new float[n];
    float* f_mid = new float[n];
    float* f_np2 = new float[n];
    float* y_mid = new float[n];
    float* y_guess = new float[n];
    float* temp = new float[n];

    ode_function(t, y_n, f_n, n);

    // Compute y_mid using trapezoidal rule
    copy(y_guess, y_n, n);
    for (int iter = 0; iter < max_iterations; ++iter) {
        ode_function(t + h, y_guess, f_mid, n);
        copy(y_mid, y_n, n);
        axpy(h / 2.0, f_n, y_mid, n);
        axpy(h / 2.0, f_mid, y_mid, n);
        copy(y_guess, y_mid, n);
    }

    // Compute y_np2 using Simpson's rule
    copy(y_guess, y_mid, n);
    for (int iter = 0; iter < max_iterations; ++iter) {
        ode_function(t + 2.0 * h, y_guess, f_np2, n);
        copy(y_np2, y_n, n);
        axpy(h / 3.0, f_n, y_np2, n);
        axpy(4.0 * h / 3.0, f_mid, y_np2, n);
        axpy(h / 3.0, f_np2, y_np2, n);
        copy(y_guess, y_np2, n);
    }
}

// Simpsons_solve to avoid reallocating results
void simpsons_solve(float t0, float tf, float h, float* y0, int n,
                    float* results, int* num_steps) {
    // Calculate number of steps (Simpson's rule advances two steps at a time)
    if (*num_steps <= 0) return;

    float* y = new float[n];
    float* y_np2 = new float[n];

    // Initialize results with y0
    copy(y, y0, n);
    copy(results, y0, n);
    float t = t0;

    // Advance using Simpson's rule
    for (int i = 1; i < *num_steps; ++i) {
        simpsons_step(t, h, y, n, y_np2);
        copy(y, y_np2, n);
        copy(results + i * n, y, n);
        t += 2.0 * h;
    }

}


void compute_metrics(float t0, float h, float* results, int n, int num_steps,
                     float* max_abs_error, float* mean_abs_error, float* rmse,
                     float* max_rel_error) {
    float* y_exact = new float[n];
    float sum_abs_error = 0.0;
    float sum_sq_error = 0.0;
    *max_abs_error = 0.0;
    *max_rel_error = 0.0;
    int total_points = num_steps * n;

    for (int i = 0; i < num_steps; ++i) {
        float t = t0 + i * 2.0 * h; // Simpson's rule uses 2h steps
        analytical_solution(t, y_exact, n);
        for (int j = 0; j < n; ++j) {
            float error = abs(results[i * n + j] - y_exact[j]);
            sum_abs_error += error;
            sum_sq_error += error * error;
            *max_abs_error = std::max(*max_abs_error, error);
            if (abs(y_exact[j]) > 1e-10) {
                float rel_error = error / abs(y_exact[j]);
                *max_rel_error = std::max(*max_rel_error, rel_error);
            }
        }
    }

    *mean_abs_error = sum_abs_error / total_points;
    *rmse = sqrt(sum_sq_error / total_points);
}


int main() {
    int n = 10;
    float t0 = 0.0;
    float tf = 1.0;
    float* y0 = new float[n];
    y0[0] = 1.0;
    for (int i = 1; i < n; ++i) y0[i] = 0.0;

    float h = 0.001;
    std::cout << "h, Max Abs Error, Mean Abs Error, RMSE, Max Rel Error\n";

    int num_steps = static_cast<int>((tf - t0) / (2.0 * h)) + 1;
    float* results = new float[num_steps * n];

    simpsons_solve(t0, tf, h, y0, n, results, &num_steps);

    float max_abs_error, mean_abs_error, rmse, max_rel_error;
    compute_metrics(t0, h, results, n, num_steps,
                    &max_abs_error, &mean_abs_error, &rmse, &max_rel_error);

    std::cout << h << ", " << max_abs_error << ", " << mean_abs_error << ", "
              << rmse << ", " << max_rel_error << "\n";


    PROMISE_CHECK_ARRAY(results, num_steps * n);
    return 0;
}

    