#include <iostream>
#include <cmath>
using namespace std;

int main() {
    __PROMISE__ a = 3.145341;
    __PROMISE__ b = 2.7635341;
    int len = 5;
    double temp = 0.0;
    __PROMISE__ max_value = (a > b) ? a / 5 : temp;

    max_value = max_value - sqrt(a);
    cout << "The maximum value is " << max_value << endl;

    PROMISE_CHECK_VAR(max_value);

    return 0;
}