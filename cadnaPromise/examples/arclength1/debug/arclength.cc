#include <stdio.h>
#include <fstream>
#include <math.h>
#include <iostream>
#include <iomanip>


float fun( float x){
  int k, n = 5;
  float t1;
  flx::floatx<5, 10> d1; d1= 1.0;

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
  float h;
  float t1;float t2;float dppi;
  float s1;
  std::ofstream res;
  std::cout.precision(15);


  t1 = -1.0;
  dppi = acos(t1);
  s1 = 0.0;
  t1 = 0.0;
  h = dppi / n;

  for ( i = 1; i <= n; i++)
    {
      t2 = fun(i * h);
      s1 = s1 + sqrt(h*h + (t2 - t1) * (t2 - t1));
      t1 = t2;
      //if (i%1000==0) 
    }


  std::cout << s1 << std::endl;
  

  return 0;

}
