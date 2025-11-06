#include <math.h>
#include <fstream>
#include <iostream>


__PROMISE__ MyFunc( __PROMISE__ x )
{
  __PROMISE__ tmp = sin(x);
  return tmp;
}

int main( )
{
   int n = 10;
   int i;
  __PROMISE__ dH = M_PI;
  __PROMISE__ dI = 0;
  double b = 0.1;
   std::ofstream res;

   res.open("result.txt");
   __PROMISE__ dX1 = 0;
  dI = dI + b;
  //  dI.display();
  for (i = 0; i < n; i++)
  {
    //   dI.display();
    //std::cout << dI << std::endl;
    dI += dH * MyFunc(dX1);
    dX1 += dH;
  }

  std::cout << std::setprecision(15)<< dI << std::endl;
   PROMISE_CHECK_VAR(dI);



  return 0;

}
