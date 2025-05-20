#include <iostream>
#include <cmath>
using namespace std;

int main() {
    float a = 3.145341;
    float b = 2.7635341;
    int len = 5;
    double x = 0.0;
    float max_value = (a > b) ? a / 5 : x;

    max_value = max_value - sqrt(a);
    cout << "The maximum value is " << max_value << endl;

    

    return 0;
}