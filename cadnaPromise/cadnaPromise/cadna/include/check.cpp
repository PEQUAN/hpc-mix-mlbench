
#include <floatx.hpp>
#include <fxmath.hpp>
#include <cmath>
#include <iostream>
#include <cassert>
#include <vector>
#include <cstdlib>  // for rand() and srand()
#include <ctime>    // for time()
#include <iomanip>
using namespace fxmath;

int main() {
    const int size = 10;  // size of the arrays

    std::vector<float> array1(size);
    std::vector<float> array2(size);
    std::vector<float> sum(size);

    std::srand(0); 
    for (int i = 0; i < size; ++i) {
        array1[i] = static_cast<float>(std::rand()) / RAND_MAX;
        array2[i] = static_cast<float>(std::rand()) / RAND_MAX;
        sum[i] = sin(array1[i] + array2[i]);
    }

    std::vector<flx::floatx<8, 23>> array1_fx(size);
    std::vector<flx::floatx<8, 23>> array2_fx(size);
    std::vector<flx::floatx<8, 23>> sum_fx(size);

    std::srand(0); 
    for (int i = 0; i < size; ++i) {
        array1_fx[i] = static_cast<flx::floatx<8, 23>>(std::rand()) / RAND_MAX;
        array2_fx[i] = static_cast<flx::floatx<8, 23>>(std::rand()) / RAND_MAX;
        sum_fx[i] = sin(array1_fx[i] + array2_fx[i]);
    }

    std::cout << std::fixed << std::setprecision(5);

    for (int i=0; i<size;i++) {
        assert((sum[i] - sum_fx[i]) ==0);
    }
    std::cout << "Test passed!\n";

    return 0;
}
