#include <iostream>

int main(){
    int a = 5;
    __PROMISE__ h = 0x1.83485f0d86984p-2;
    __PROMISE__ b = 0x1.0d8f1c70e22f7p-68;

    __PROMISE__ c = (__PROMISE__)a + h + b - (__PROMISE__)a;
    PROMISE_CHECK_VAR(c);
    std::cout << c << std::endl;
    return 0;
}