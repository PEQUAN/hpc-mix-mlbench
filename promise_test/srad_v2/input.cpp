// perl cadnaizer -o output.c -d input.c

double cal_double(float a, float b){
    return a + b;
}



double cal_float(float a, float b){
    return a + b;
}

int main() {
    float a = 3.f;
    float b = a + 4.0;
    double c = cal_float(a, b);
    return 0;
}
