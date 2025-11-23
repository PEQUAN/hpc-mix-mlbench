#include <half.hpp>
#include <floatx.hpp>
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



void ode_function(flx::floatx<5, 2> t, float* y, float* dydt, int n) {
    // ODE: dy_i/dt = y_{i-1} - 2*y_i + y_{i+1} (tridiagonal system)
    dydt[0] = -2.0 * y[0] + y[1];
    for (int i = 1; i < n - 1; ++i) {
        dydt[i] = y[i - 1] - 2.0 * y[i] + y[i + 1];
    }
    dydt[n - 1] = y[n - 2] - 2.0 * y[n - 1];
}


void analytical_solution(flx::floatx<5, 2> t, float* y_exact, int n) {
    // Analytical solution: for n=2, exact; for n>2, use RK4 with small h
    if (n == 2) {
        y_exact[0] = 0.5 * exp(-t) + 0.5 * exp(-3.0 * t);
        y_exact[1] = 0.5 * exp(-t) - 0.5 * exp(-3.0 * t);
    } else {
        // Use RK4 with small h (h_ref = 0.0001) as reference
        flx::floatx<5, 2> t0 = 0.0;
        flx::floatx<5, 2> h_ref = 0.0001;
        int steps = static_cast<int>(t / h_ref) + 1;
        float* y = new float[n];
        float* y_new = new float[n];
        y[0] = 1.0;
        for (int i = 1; i < n; ++i) y[i] = 0.0;
        flx::floatx<5, 2> current_t = t0;
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

void trapezoidal_step(flx::floatx<5, 2> t, float h, float* y, int n, float* y_new) {

    // Trapezoidal Rule step with fixed-point iteration
    int max_iterations = 5; // Fixed number of iterations
    float* f_n = new float[n];
    float* f_np1 = new float[n];
    float* y_guess = new float[n];
    float* temp = new float[n];

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


void trapezoidal_solve(flx::floatx<5, 2> t0, flx::floatx<5, 2> tf, float h, float* y0, int n,
                       float* results, int* num_steps) {
    if (*num_steps <= 0) return;

    float* y = new float[n];
    float* y_new = new float[n];

    copy(y, y0, n);
    copy(results, y0, n);
    flx::floatx<5, 2> t = t0;

    for (int i = 1; i < *num_steps; ++i) {
        trapezoidal_step(t, h, y, n, y_new);
        copy(y, y_new, n);
        copy(results + i * n, y, n);
        t += h;
    }

    delete[] y;
    delete[] y_new;
}


int main() {
    int n = 10;
    flx::floatx<5, 2> t0 = 0.0;
    flx::floatx<5, 2> tf = 1.0;
    float* y0 = new float[n];
    y0[0] = 1.0;
    for (int i = 1; i < n; ++i) y0[i] = 0.0;


    float h = 0.001;
    std::cout << "h, Max Abs Error, Mean Abs Error, RMSE, Max Rel Error\n";

    int num_steps = static_cast<int>((tf - t0) / h) + 1;
    float* results = new float[num_steps * n];

    trapezoidal_solve(t0, tf, h, y0, n, results, &num_steps);

    PROMISE_CHECK_ARRAY(results, num_steps * n);
    delete[] results;

    delete[] y0;
    return 0;
}
    