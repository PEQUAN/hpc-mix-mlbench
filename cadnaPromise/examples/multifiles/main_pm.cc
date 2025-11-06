#include <iostream>
#include "callfunc_pm.h"

int main(int argc, char **argv){
    __PR_1__ arg1 = atof(argv[1]);
    __PR_1__ arg2 = atof(argv[2]);
    __PR_1__ c = sum(arg1, arg2);
    PROMISE_CHECK_VAR(c);
    std::cout << c << std::endl;
    return 0;
}