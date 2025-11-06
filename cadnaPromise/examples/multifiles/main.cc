#include <iostream>
#include "callfunc.h"

int main(int argc, char **argv){
    float arg1 = atof(argv[1]);
    float arg2 = atof(argv[2]);
    float c = sum(arg1, arg2);
    std::cout << c << std::endl;
    return 0;
}