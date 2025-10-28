#include <cadna.h>
using namespace std;

// perl cadnaizer -o output.c -d input.c

double_st cal_double(float_st a, float_st b){
    return a + b;
}



double_st cal_float(float_st a, float_st b){
    return a + b;
}

int main() {
    cadna_init(0);

    float_st a = 3.f;
    float_st b = a + 4.0;
    double_st c = cal_float(a, b);
    return 0;

    cadna_end();
}
