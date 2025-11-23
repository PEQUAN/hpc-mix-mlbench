#include <half.hpp>
#include <floatx.hpp>
#include <iostream>
#include <cmath>

float true_function(double x) {
    // True function for generating data points and exact values
    return sin(x);
}


float lagrange_interpolate(float x_eval, double* x, double* y, int n) {
    // Lagrange interpolation at x_eval
    float result = 0.0;
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


int main() {
    int n= 20;
    int num_eval = 100; 
    flx::floatx<4, 3> x_start = 0.0;
    float x_end = 4.5;

    std::cout << "n, Max Abs Error, Mean Abs Error, RMSE, Max Rel Error\n";

    // Generate data points
    double* x = new double[n];
    double* y = new double[n];
    double h = (x_end - x_start) / (n - 1);
    for (int i = 0; i < n; ++i) {
        x[i] = x_start + i * h;
        y[i] = true_function(x[i]);
    }

    float* x_eval = new float[num_eval];  // Evaluate Lagrange interpolation on fine grid
    float* y_interp = new float[num_eval];
    float h_eval = (x_end - x_start) / (num_eval - 1);

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