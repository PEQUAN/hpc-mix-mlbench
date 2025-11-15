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


void ode_function(__PROMISE__ t, __PROMISE__* y, __PROMISE__* dydt, int n) {
    // ODE: dy_i/dt = y_{i-1} - 2*y_i + y_{i+1} (tridiagonal system)
    dydt[0] = -2.0 * y[0] + y[1];
    for (int i = 1; i < n - 1; ++i) {
        dydt[i] = y[i - 1] - 2.0 * y[i] + y[i + 1];
    }
    dydt[n - 1] = y[n - 2] - 2.0 * y[n - 1];
}

void analytical_solution(__PROMISE__ t, __PROMISE__* y_exact, int n) {
    // Analytical solution: for n=2, exact; for n>2, use RK4 with small h
    if (n == 2) {
        y_exact[0] = 0.5 * exp(-t) + 0.5 * exp(-3.0 * t);
        y_exact[1] = 0.5 * exp(-t) - 0.5 * exp(-3.0 * t);
    } else {
        // Use RK4 with small h (h_ref = 0.0001) as reference
        __PROMISE__ t0 = 0.0;
        __PROMISE__ h_ref = 0.0001;
        int steps = static_cast<int>(t / h_ref) + 1;
        __PROMISE__* y = new __PROMISE__[n];
        __PROMISE__* y_new = new __PROMISE__[n];
        y[0] = 1.0;
        for (int i = 1; i < n; ++i) y[i] = 0.0;
        __PROMISE__ current_t = t0;
        for (int i = 0; i < steps && current_t < t; ++i) {
            // RK4 step for reference solution
            __PROMISE__* k1 = new __PROMISE__[n];
            __PROMISE__* k2 = new __PROMISE__[n];
            __PROMISE__* k3 = new __PROMISE__[n];
            __PROMISE__* k4 = new __PROMISE__[n];
            __PROMISE__* temp = new __PROMISE__[n];

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
void simpsons_step(__PROMISE__ t, __PROMISE__ h, __PROMISE__* y_n, int n, __PROMISE__* y_np2) {
    int max_iterations = 5;
    __PROMISE__* f_n = new __PROMISE__[n];
    __PROMISE__* f_mid = new __PROMISE__[n];
    __PROMISE__* f_np2 = new __PROMISE__[n];
    __PROMISE__* y_mid = new __PROMISE__[n];
    __PROMISE__* y_guess = new __PROMISE__[n];
    __PROMISE__* temp = new __PROMISE__[n];

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
void simpsons_solve(__PROMISE__ t0, __PROMISE__ tf, __PROMISE__ h, __PROMISE__* y0, int n,
                    __PROMISE__* results, int* num_steps) {
    // Calculate number of steps (Simpson's rule advances two steps at a time)
    if (*num_steps <= 0) return;

    __PROMISE__* y = new __PROMISE__[n];
    __PROMISE__* y_np2 = new __PROMISE__[n];

    // Initialize results with y0
    copy(y, y0, n);
    copy(results, y0, n);
    __PROMISE__ t = t0;

    // Advance using Simpson's rule
    for (int i = 1; i < *num_steps; ++i) {
        simpsons_step(t, h, y, n, y_np2);
        copy(y, y_np2, n);
        copy(results + i * n, y, n);
        t += 2.0 * h;
    }

}




int main() {
    int n = 10;
    __PROMISE__ t0 = 0.0;
    __PROMISE__ tf = 1.0;
    __PR_1__* y0 = new __PR_1__[n];
    y0[0] = 1.0;
    for (int i = 1; i < n; ++i) y0[i] = 0.0;

    __PROMISE__ h = 0.001;
    std::cout << "h, Max Abs Error, Mean Abs Error, RMSE, Max Rel Error\n";

    int num_steps = static_cast<int>((tf - t0) / (2.0 * h)) + 1;
    __PR_2__* results = new __PR_2__[num_steps * n];

    simpsons_solve(t0, tf, h, y0, n, results, &num_steps);


    PROMISE_CHECK_ARRAY(results, num_steps * n);
    return 0;
}

    