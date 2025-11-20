#include <half.hpp>
#include <floatx.hpp>
#include <iostream>

void sumArrays(float* arr1, float* arr2, flx::floatx<5, 5>* result, half_float::half *sum, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = arr1[i] - arr2[i];
        *sum += (double)result[i];
    }
}

int main() {
    int size = 5;
    float* arr1 = new float[size] {1.112, 2.2392, 3.315, 4.436, 5.5};
    float* arr2 = new float[size] {6.63, 7.717, 8.82, 9.9, 10.141};
    flx::floatx<5, 5>* result = new flx::floatx<5, 5>[size];
    half_float::half result_sum; 

    sumArrays(arr1, arr2, result, &result_sum, size);
    
    std::cout << std::endl;
    delete[] arr1; delete[] arr2; delete[] result;

    return 0;
}