#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>

__PROMISE__ x_next(__PROMISE__ a, __PROMISE__ x)
{
  __PROMISE__ tmp = (a/x);
  return ((x+tmp)/2);
}

__PROMISE__ squareRoot(__PROMISE__ a)
{
  __PROMISE__ x = 1.;
  for(int i = 0; i < 10; i++)
    x = x_next(a, x);

  return x;
}


int main()
{
  __PROMISE__ x;
  x = squareRoot(2);

  std::cout <<std::setprecision(15) << x << std::endl;
  PROMISE_CHECK_VAR(x);


  return 0;


}
