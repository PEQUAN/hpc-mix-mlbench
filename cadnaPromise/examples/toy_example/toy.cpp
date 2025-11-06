#include <iostream>

void sumArrays(__PR_1__* arr1, __PR_2__* arr2, __PR_3__* result, __PROMISE__ *sum, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = arr1[i] - arr2[i];
        *sum += (double)result[i];
    }
}

int main() {
    int size = 5;
    __PR_1__* arr1 = new __PR_1__[size] {1.112, 2.2392, 3.315, 4.436, 5.5};
    __PR_2__* arr2 = new __PR_2__[size] {6.63, 7.717, 8.82, 9.9, 10.141};
    __PR_3__* result = new __PR_3__[size];
    __PROMISE__ result_sum; 

    sumArrays(arr1, arr2, result, &result_sum, size);
    PROMISE_CHECK_VAR(result_sum);
    std::cout << std::endl;
    delete[] arr1; delete[] arr2; delete[] result;

    return 0;
}