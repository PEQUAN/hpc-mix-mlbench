#include <iostream>

int main(){
    int a = 5;
    double h; h= 0x1.83485f0d86984p-2;
    half_float::half b; b= 0x1.0d8f1c70e22f7p-68;

    double c = (half_float::half)a + h + b - (half_float::half)a;
    
    std::cout << c << std::endl;
    return 0;
}