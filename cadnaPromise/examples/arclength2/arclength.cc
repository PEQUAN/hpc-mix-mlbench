#include <stdio.h>
#include <fstream>
#include <math.h>
#include <iostream>
#include <iomanip>


__PR_fun__ fun( __PR_1__ x){
  int k, n = 5;
  __PR_fun__ t1;
  __PR_d1__ d1 = 1.0;

  t1 = x;
  for ( k = 1; k <= n; k++ )
    {
      d1 = 2.0 * d1;
      t1 = t1+ sin(d1 * x)/d1;
    }
  return t1;
}

int main( int argc, char **argv) {


  int i,n = 1000000;
  __PR_1__ h;
  __PR_fun__ t1, t2;
  __PROMISE__ s1, dppi;
  std::ofstream res;
  std::cout.precision(15);


  t1 = -1.0;
  dppi = acos(t1);
  s1 = 0.0;
  t1 = 0.0;
  h = dppi / n;

  for ( i = 1; i <= n; i++)
    {
      t2 = fun((__PR_1__)i * h);
      s1 = s1 + sqrt(h*h + (t2 - t1) * (t2 - t1));
      t1 = t2;
      //if (i%1000==0) PROMISE_CHECK_VAR(t1);
    }


  std::cout << s1 << std::endl;
  PROMISE_CHECK_VAR(s1);

  return 0;

}
