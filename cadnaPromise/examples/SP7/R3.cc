#include <iostream>



void dSS(__PROMISE__ y[3], __PROMISE__ u[3],__PROMISE__ xk[5])
{
	__PROMISE__ x0_kp1;
    __PROMISE__ x1_kp1;
    __PROMISE__ x2_kp1;
    __PROMISE__ x3_kp1;
    __PROMISE__ x4_kp1;

    
    
    __PROMISE__ c1[6];
    c1[0] = 0x1.199999999999ap+0;
    c1[1] = 0x1.599999999999ap+1;
    c1[2] = 0x1.ccccccccccccdp-1;
    c1[3] = 0x1.999999999999ap-2;
    c1[4] = 0x1.8000000000000p+0;
    c1[5] = 0x1.999999999999ap-1;

    
    __PROMISE__ c2[7];
    c2[0] = 0x1.0cccccccccccdp+1;
    c2[1] = 0x1.8cccccccccccdp+1;
    c2[2] = 0x1.3333333333333p-2;
    c2[3] = 0x1.999999999999ap-3;
    c2[4] = 0x1.999999999999ap-4;
    c2[5] = 0x1.3333333333333p-2;
    c2[6] = 0x1.3333333333333p-1;
    
    __PROMISE__ c3[7];
    c3[0] = 0x1.599999999999ap+2;
    c3[1] = 0x1.999999999999ap+0;
    c3[2] = -0x1.b333333333333p+0;
    c3[3] = -0x1.a666666666666p+2;
    c3[4] = 0x1.8000000000000p+1;
    c3[5] = 0x1.0000000000000p-1;
    c3[6] = 0x1.999999999999ap-2;
    
    __PROMISE__ c4[2];
    c4[0] = 0x1.26e978d4fdf3bp-4;
    c4[1] = 0x1.8000000000000p+0;

    __PROMISE__ c5[2];
    c5[0] = 0x1.3333333333333p-2;
    c5[1] = 0x1.999999999999ap-3;
    
    __PROMISE__ c6[2];
    c6[0] = -0x1.999999999999ap-4;
    c6[1] = 0x1.ccccccccccccdp-1;
    
    __PROMISE__ c7;
    c7 = 0x1.999999999999ap-5;
    
    __PROMISE__ c8;
    c8 = 0x1.999999999999ap-2;
	// intermediate variable(s)


	//output(s)
	y[0]  = c1[0]*xk[0] +
            c1[1]*xk[1] +  
            c1[2]*xk[2] +
            c1[3]*xk[3] +  
            c1[4]*xk[4] +
            u[0] + c1[5]*u[1];


	y[1]  = c2[0]*xk[0] +
            c2[1]*xk[1] +  
            c2[2]*xk[2] +
            c2[3]*xk[3] + 
            c2[4]*xk[4] +
            c2[5]*u[0] + 
            c2[6]*u[1];

	y[2]  = c3[0]*xk[0] +
            c3[1]*xk[1] + 
            c3[2]*xk[2] +
            c3[3]*xk[3] + 
            c3[4]*xk[4] +
            c3[5]*u[0] + 
            c3[6]*u[1];


	//states
	x0_kp1  = c4[0]*xk[2] +
              c4[1]*xk[4] + u[0];

	x1_kp1  = xk[0] + c5[0]*xk[2] +
              c5[1]*xk[4];

	x2_kp1  = xk[1] + c6[0]*xk[2] +
              c6[1]*xk[4];

	x3_kp1  = c7*xk[4] + u[1];
	x4_kp1  = xk[3] + c8*xk[4];


	//permutations
	xk[0] = x0_kp1;
	xk[1] = x1_kp1;
	xk[2] = x2_kp1;
	xk[3] = x3_kp1;
	xk[4] = x4_kp1;



}

int seed = 1234;
int rand2(){
    seed = (16807 * seed) % 2147483647;
    return 21878754 * (seed / 2147483647);
}

int main() {
  __PROMISE__ z[3]={};  
  __PROMISE__ u[2] ;
  __PROMISE__ xk[5] = {};


 for (int i=0; i<=99; i++){
  u[0] = rand2() % (1 + 1 + 1) - 1;
  u[1] = rand2() % (1 + 1 + 1) - 1;
  dSS(z,u,xk);  

  }  

  PROMISE_CHECK_ARRAY(z,3);
  for(int i=0; i<3; i++)
  std::cout<<z[i]<<std::endl;

    return 0;

}
