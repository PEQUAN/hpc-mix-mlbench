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

void rk4_step(__PROMISE__ t, __PROMISE__ h, __PROMISE__* y, int n, __PROMISE__* y_new) {
    // RK4 step
    __PROMISE__* k1 = new __PROMISE__[n];
    __PROMISE__* k2 = new __PROMISE__[n];
    __PROMISE__* k3 = new __PROMISE__[n];
    __PROMISE__* k4 = new __PROMISE__[n];
    __PROMISE__* temp = new __PROMISE__[n];

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


// Analytical solution: for n=2, exact; for n>2, use RK4 with small h
void analytical_solution(__PROMISE__ t, __PROMISE__* y_exact, int n) {
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
            rk4_step(current_t, h_ref, y, n, y_new);
            copy(y, y_new, n);
            current_t += h_ref;
        }
        copy(y_exact, y, n);
        delete[] y;
        delete[] y_new;
    }
}

// RK4 solver
void rk4_solve(__PROMISE__ t0, __PROMISE__ tf, __PROMISE__ h, __PROMISE__* y0, int n,
               __PROMISE__* results, int* num_steps) {

    if (*num_steps <= 0) return;

    __PROMISE__* y = new __PROMISE__[n];
    __PROMISE__* y_new = new __PROMISE__[n];

    copy(y, y0, n);
    copy(results, y0, n);
    __PROMISE__ t = t0;

    for (int i = 1; i < *num_steps; ++i) {
        rk4_step(t, h, y, n, y_new);
        copy(y, y_new, n);
        copy(results + i * n, y, n);
        t += h;
    }

    delete[] y;
    delete[] y_new;
}



int main() {
    // Problem setup
    int n = 10; // Changed to n=10
    __PROMISE__ t0 = 0.0;
    __PROMISE__ tf = 1.0;
    __PR_1__* y0 = new  __PR_1__[n];
    y0[0] = 1.0;
    for (int i = 1; i < n; ++i) y0[i] = 0.0;

    __PROMISE__ h = 0.001; // Test different step sizes

    std::cout << "h, Max Abs Error, Mean Abs Error, RMSE, Max Rel Error\n";
    int num_steps = static_cast<int>((tf - t0) / h) + 1;
     __PR_2__* results = new  __PR_2__[num_steps * n];
 
    rk4_solve(t0, tf, h, y0, n, results, &num_steps);

    PROMISE_CHECK_ARRAY(results, num_steps * n);
    delete[] results;
  
    delete[] y0;
    return 0;
}