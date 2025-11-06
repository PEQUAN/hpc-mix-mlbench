#include <half_promise.hpp>
#include <fxmath.hpp>
#include <cmath>
#include <iostream>
using namespace fxmath;

int main(){
    half_float::half x(0.412312);
    std::cout << sin(x) << std::endl;
}