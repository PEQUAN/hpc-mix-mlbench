#include <iostream>

double sumVariables(double a, double b) {
    return a + b;
}

void sumArrays(double* arr1, double* arr2, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = arr1[i] + arr2[i];
    }
}

int main() {
    int size = 5;
    double* arr1 = new double[size] {1.112, 2.2392, 3.315, 4.436, 5.5};
    double* arr2 = new double[size] {6.63, 7.717, 8.82, 9.9, 10.141};
    double* result = new double[size];

    double var1 = 15.5, var2 = 26.7212;
    

    sumArrays(arr1, arr2, result, size);
    
    double result_var = sumVariables(var1, var2);
    std::cout << "Sum of variables: " << result_var << std::endl;

    std::cout << "Sum of arrays: ";
    for (int i = 0; i < size; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;

    delete[] arr1;
    delete[] arr2;
    delete[] result;

    return 0;
}