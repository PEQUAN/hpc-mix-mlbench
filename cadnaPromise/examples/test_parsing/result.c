/* non-sens code to test the right parsing of the types __PROMISE__ and __PR_xxx__, the variable declarations, etc. */
#include <stdio.h>

double foo(double bar, float foobar){
    return bar*foobar;
}

double add_function(double alpha2, float alpha3){
    return alpha2+alpha3;
}

int main() {
    double t; t= 12;
    float x;float y; y=0;float z[24];   /* x, y and z are __PR_xyZ__ */
    double c1; c1=5;double c2;      /* but here c1 and c2 are two different types */
    double *a;double b;
    double *bar; bar= NULL;
    double xx; xx=pow(2,N);
    double zz[12]= {0};
    /*__PROMISE__ z;*/
    // __PROMISE__ x
    t = x+y + foo(c1, x);

    printf("toto__PROMISE__\n");
  return 0;
}
