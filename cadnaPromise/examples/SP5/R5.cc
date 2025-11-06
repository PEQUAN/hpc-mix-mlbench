#include <iostream>



__PROMISE__ rhoDFII(__PROMISE__ u, __PROMISE__ xk[15])
{
	__PROMISE__ y;

	// intermediate variable(s)
	__PROMISE__ T0;
    T0 = xk[0] + 0x1.ec0c0a1a09328p-43*u;


	//output(s)
	y  = T0;

    __PROMISE__ c0;
    c0 = -0x1.80a82bc083b28p+1;
    __PROMISE__ c1;
    c1 = 0x1.cd4b4978689f6p-38;
    __PROMISE__ c2;
    c2 = -0x1.1dcb423001600p+2;
    __PROMISE__ c3;
    c3 = 0x1.93a1e0495b8b7p-34;
    __PROMISE__ c4;
    c4 = -0x1.15e3583e3de00p+2;
    __PROMISE__ c5;
    c5 = 0x1.b544b2fa232c6p-31;
    __PROMISE__ c6;
    c6 = -0x1.8aa6416b2c200p+1;
    __PROMISE__ c7;
    c7 = 0x1.47f3863b9a615p-28;
    __PROMISE__ c8;
    c8 = -0x1.b0d0f3f807000p+0;
    __PROMISE__ c9;
    c9 = 0x1.68bf13a7f69e4p-26;
    __PROMISE__ c10;
    c10 = -0x1.7a21d50950000p-1;
    __PROMISE__ c11;
    c11 = 0x1.2c9f3b0bf82e8p-24;
    __PROMISE__ c12;
    c12 = -0x1.0ba7e13f80000p-2;
    __PROMISE__ c13;
    c13 = 0x1.8283950f63a98p-23;
    __PROMISE__ c14;
    c14 = -0x1.353630ab58000p-4;
    __PROMISE__ c15;
    c15 = 0x1.8283950f63a99p-22;
    __PROMISE__ c16;
    c16 = -0x1.234463e800000p-6;
    __PROMISE__ c17;
    c17 = 0x1.2c9f3b0bf82e9p-21;
    __PROMISE__ c18;
    c18 = -0x1.bb7752d800000p-9;
    __PROMISE__ c19;
    c19 = 0x1.68bf13a7f69e4p-21;
    __PROMISE__ c20;
    c20 = -0x1.0bcaca0000000p-11;
    __PROMISE__ c21;
    c21 = 0x1.47f3863b9a614p-21;
    __PROMISE__ c22;
    c22 = -0x1.f075b63f00000p-15;
    __PROMISE__ c23;
    c23 = 0x1.b544b2fa232c6p-22;
    __PROMISE__ c24;
    c24 = -0x1.4d569aa000000p-18;
    __PROMISE__ c25;
    c25 = 0x1.93a1e0495b8b6p-23;
    __PROMISE__ c26;
    c26 = -0x1.21daa90000000p-22;
    __PROMISE__ c27;
    c27 = 0x1.cd4b4978689f4p-25;
    __PROMISE__ c28;
    c28 = -0x1.ec08800000000p-28;
    __PROMISE__ c29;
    c29 = 0x1.ec0c0a1a09328p-28;
	//states
	xk[0]  = c0*T0 + xk[0] + xk[1] + c1*u;
	xk[1]  = c2*T0 + xk[1] + xk[2] + c3*u;
	xk[2]  = c4*T0 + xk[2] + xk[3] + c5*u;
	xk[3]  = c6*T0 + xk[3] + xk[4] + c7*u;
	xk[4]  = c8*T0 + xk[4] + xk[5] + c9*u;
	xk[5]  = c10*T0 + xk[5] + xk[6] + c11*u;
	xk[6]  = c12*T0 + xk[6] + xk[7] + c13*u;
	xk[7]  = c14*T0 + xk[7] + xk[8] + c15*u;
	xk[8]  = c16*T0 + xk[8] + xk[9] + c17*u;
	xk[9]  = c18*T0 + xk[9] + xk[10] + c19*u;
	xk[10]  = c20*T0 + xk[10] + xk[11] + c21*u;
	xk[11]  = c22*T0 + xk[11] + xk[12] + c23*u;
	xk[12]  = c24*T0 + xk[12] + xk[13] + c25*u;
	xk[13]  = c26*T0 + xk[13] + xk[14] + c27*u;
	xk[14]  = c28*T0 + xk[14] + c29*u;


	return y;
}

int seed = 1234;
int rand2(){
    seed = (16807 * seed) % 2147483647;
    return 21878754 * (seed / 2147483647);
}

int main(){
  __PROMISE__ z;  
  __PROMISE__ u ;
  __PROMISE__ xk[15]={};
  
 for (int i=0; i<=99; i++){
   u = rand2() % (1 + 1 + 1) - 1;
   z= rhoDFII(u,xk);  
     
  }  
  PROMISE_CHECK_VAR(z);
  std::cout<<z<<std::endl;
  return 0;
    
}
