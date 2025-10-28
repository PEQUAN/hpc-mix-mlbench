#include <iostream>


int main(){


    double a = 3;

    double b = 2.9999999999999999999999999;
    double c = a - b;

    c = a + b;
    b = c;
    std::cout << "c:" << b << std::endl;
    return 0;
}