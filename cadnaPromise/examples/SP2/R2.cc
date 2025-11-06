#include <iostream>


__PROMISE__ dSS(__PROMISE__ u, __PROMISE__ xk[4])
{
	__PROMISE__ x0_kp1;
    __PROMISE__ x1_kp1;
    __PROMISE__ x2_kp1;
    __PROMISE__ x3_kp1;
	__PROMISE__ y;

	// intermediate variable(s)
    __PROMISE__ c1[5];
    
    c1[0] = 0x1.37b33fa593f80p-12;
    c1[1] = 0x1.3ecac7c0dd588p-57;
    c1[2] = -0x1.3d49ddc32f4f6p-8;
    c1[3] = 0x1.3ecac7c0dd588p-57;
    c1[4] = 0x1.0c74de1bacec4p-7;

    
    __PROMISE__ c2[4];
    
    c2[0] = -0x1.ed6c34a4458adp-1;
    c2[1] = 0x1.1800000000000p-49;
    c2[2] = -0x1.f18d8efdba52ap+0;
    c2[3] = 0x1.1800000000000p-49;

	//output(s)
	y  = c1[0]*xk[0] +
         c1[1]*xk[1] + 
         c1[2]*xk[2] +
         c1[3]*xk[3] + 
         c1[4]*u;


	//states
	x0_kp1  = xk[1];
	x1_kp1  = xk[2];
	x2_kp1  = xk[3];
	x3_kp1  = c2[0]*xk[0] +
              c2[1]*xk[1] + 
              c2[2]*xk[2] +
              c2[3]*xk[3] + u;


	//permutations
	xk[0] = x0_kp1;
	xk[1] = x1_kp1;
	xk[2] = x2_kp1;
	xk[3] = x3_kp1;


	return y;
}

int seed = 1234;
int rand2(){
    seed = (16807 * seed) % 2147483647;
    return 21878754 * (seed / 2147483647);
}

int main() {
    
  __PROMISE__ z;  
  __PROMISE__ u ;
  __PROMISE__ xk[4] = {} ;

  

 for (int i=0; i<=99; i++){
   u = rand2() % (1 + 1 + 1) - 1;
   z= dSS(u,xk);  

     

  }  

  PROMISE_CHECK_VAR(z);

  std::cout<<z<<std::endl;
    
    
    return 0;
    
}
