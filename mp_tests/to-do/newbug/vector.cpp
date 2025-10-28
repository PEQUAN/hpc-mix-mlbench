#include <iostream>
#include <vector>
#include <random>

std::vector<__PROMISE__> vectorSubtraction() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<__PROMISE__> dist(0.0, 100.0);
    
    std::vector<__PROMISE__> vec1(5);
    std::vector<__PROMISE__> vec2(5);
    
    for(size_t i = 0; i < 5; ++i) {
        vec1[i] = dist(rng);
        vec2[i] = dist(rng);
    }
    
    std::vector<__PROMISE__> result(5);
    for(size_t i = 0; i < 5; ++i) {
        result[i] = vec1[i] - vec2[i];
    }
    
    return result;
}

int main() {
    std::vector<__PROMISE__> result = vectorSubtraction();
    
    std::cout << "Result of vector subtraction:\n";
    for(__PROMISE__ val : result) {
        std::cout << val << " ";
    }


    float result2[5];
    // double* result2 = new double[5]; works!
    // double* result2 = (double*) malloc(5 * sizeof(double));



    for (int i=0; i<5; i++){
        result2[i] = result[i];
    }

    PROMISE_CHECK_ARRAY(result.data(), 5);
    std::cout << "\n";
    
    return 0;
}

