/* non-sens code to test the right parsing of the types __PROMISE__ and __PR_xxx__, the variable declarations, etc. */
#include <stdio.h>

__PROMISE__ foo(__PROMISE__ bar, __PR_xyz__ foobar){
    return bar*foobar;
}

__PROMISE__ add_function(__PROMISE__ alpha2, float alpha3){
    return alpha2+alpha3;
}

int main() {
    __PROMISE__ t = 12;
    __PR_xyz__ x,y=0,z[24];   /* x, y and z are __PR_xyZ__ */
    __PROMISE__ c1=5,c2;      /* but here c1 and c2 are two different types */
    __PROMISE__  *a,b;
    __PR_foo__ *bar = NULL;
    __PROMISE__ xx=pow(2,N);
    __PR_foo2__ zz[12] = {0};
    /*__PROMISE__ z;*/
    // __PROMISE__ x
    t = x+y + foo(c1, x);
PROMISE_CHECK_VAR(t);
    printf("toto__PROMISE__\n");
  return 0;
}
