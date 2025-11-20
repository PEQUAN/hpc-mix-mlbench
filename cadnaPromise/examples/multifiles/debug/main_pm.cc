#include <iostream>
#include "callfunc_pm.h"

int main(int argc, char **argv){
    flx::floatx<8, 23> arg1; arg1= atof(argv[1]);
    flx::floatx<8, 23> arg2; arg2= atof(argv[2]);
    flx::floatx<8, 23> c; c= sum(arg1, arg2);
    
    std::cout << c << std::endl;
    return 0;
}