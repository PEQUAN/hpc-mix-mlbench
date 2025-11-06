#include <iostream>


using namespace std;

__PROMISE__  dSS(__PROMISE__  u, __PROMISE__ *xk)
{
	__PROMISE__  x0_kp1;
    __PROMISE__  x1_kp1;
    __PROMISE__  x2_kp1;
    __PROMISE__  x3_kp1;
    __PROMISE__  x4_kp1;
    __PROMISE__  x5_kp1;
    __PROMISE__  x6_kp1;
    __PROMISE__  x7_kp1;
    __PROMISE__  x8_kp1;
    __PROMISE__  x9_kp1;
    __PROMISE__  x10_kp1;
    __PROMISE__  x11_kp1;
	__PROMISE__  y;

	// intermediate variable(s)
    __PROMISE__ c1[13];
    
    c1[0] = 0x1.0d8f1c70e22f7p-68;
    c1[1] = -0x1.64a59f636d505p-66;
    c1[2] = 0x1.4d2d4c5789c8fp-55;
    c1[3] = -0x1.b8619527c1c3bp-53;
    c1[4] = 0x1.9b0c2cc09110fp-42;
    c1[5] = -0x1.0fe3286217150p-39;
    c1[6] = 0x1.fbf1fe7be5d8ap-29;
    c1[7] = -0x1.4fbc85900b181p-26;
    c1[8] = 0x1.39758dd247fbcp-15;
    c1[9] = -0x1.9e945fa9ab96cp-13;
    c1[10] = 0x1.832f053aa95b6p-2;
    c1[11] = -0x1.fff37c15b8516p+0;
    c1[12] = 0x1.20d7651c1ed4cp-80;
    
    
    __PROMISE__ c2[3];
    
    c2[0] = 0x1.82f79cfdf224bp-2;
    c2[1] = -0x1.fff94addf28bfp-1;
    c2[2] = 0x1.9ea5ed975858dp-14;
    
    
    __PROMISE__ c3[5];
    
    c3[0] = 0x1.3963a04f6f9a4p-15;
    c3[1] = -0x1.9ea33645d5c1dp-13;
    c3[2] = 0x1.8359f00910837p-2;
    c3[3] = -0x1.fff94aee1ea71p-1;
    c3[4] = 0x1.4fce8c00d1ac6p-27;
    
    
    __PROMISE__ c4[7];
    
    c4[0] = 0x1.fb947b2c02a67p-29;
    c4[1] = -0x1.4fc889b0987ecp-26;
    c4[2] = 0x1.39afb24bca8e6p-15;
    c4[3] = -0x1.9e9e8222a9f4cp-13;
    c4[4] = 0x1.8300655b6d6bep-2;
    c4[5] = -0x1.ffedacefb78f7p-1;
    c4[6] = 0x1.0ff1c14ce880fp-40;
    
    
    __PROMISE__ c5[9];
    
    c5[0] = 0x1.9b0ce4b035397p-42;
    c5[1] = -0x1.0fece389a3978p-39;
    c5[2] = 0x1.fc0fb02a7baedp-29;
    c5[3] = -0x1.4fc4ba8cfc077p-26;
    c5[4] = 0x1.39672ef73e595p-15;
    c5[5] = -0x1.9e99cdee11542p-13;
    c5[6] = 0x1.83485f0d86984p-2;
    c5[7] = -0x1.ffedad1009cf0p-1;
    c5[8] = 0x1.b8743a9dd86c5p-54;


    __PROMISE__ c6[11];
    
    c6[0] = 0x1.4cdeae7254715p-55;
    c6[1] = -0x1.b869768181931p-53;
    c6[2] = 0x1.9b6df983cbae5p-42;
    c6[3] = -0x1.0fe805f4e9d8dp-39;
    c6[4] = 0x1.fb96eb10e6353p-29;
    c6[5] = -0x1.4fbeb86e82a9cp-26;
    c6[6] = 0x1.399f6ab2ee566p-15;
    c6[7] = -0x1.9e9716dc677f1p-13;
    c6[8] = 0x1.8314ad1f57e47p-2;
    c6[9] = -0x1.ffe6f81b44783p-1;
    c6[10] = 0x1.64ae574b994f2p-67;
    
    
    __PROMISE__ c7[13];
    
    c7[0] = 0x1.0d8f1c70e22f7p-68;
    c7[1] = -0x1.64a59f636d505p-66;
    c7[2] = 0x1.4d2d4c5789c8fp-55;
    c7[3] = -0x1.b8619527c1c3bp-53;
    c7[4] = 0x1.9b0c2cc09110fp-42;
    c7[5] = -0x1.0fe3286217150p-39;
    c7[6] = 0x1.fbf1fe7be5d8ap-29;
    c7[7] = -0x1.4fbc85900b181p-26;
    c7[8] = 0x1.39758dd247fbcp-15;
    c7[9] = -0x1.9e945fa9ab96cp-13;
    c7[10] = 0x1.832f053aa95b6p-2;
    c7[11] = -0x1.ffe6f82b70a2bp-1;
    c7[12] = 0x1.20d7651c1ed4cp-80;
    
    
	//output(s)
	y  = c1[0]*xk[0] +
         c1[1]*xk[1] + 
         c1[2]*xk[2] +
         c1[3]*xk[3] + 
         c1[4]*xk[4] +
         c1[5]*xk[5] + 
         c1[6]*xk[6] +
         c1[7]*xk[7] + 
         c1[8]*xk[8] +
         c1[9]*xk[9] + 
         c1[10]*xk[10] +
         c1[11]*xk[11] + 
         c1[12]*u;


	//states
	x0_kp1  = c2[0]*xk[0] +
              c2[1]*xk[1] + c2[2]*u;


	x1_kp1  = xk[0];
    
	x2_kp1  = c3[0]*xk[0] +
              c3[1]*xk[1] + 
              c3[2]*xk[2] +
              c3[3]*xk[3] + c3[4]*u;

	x3_kp1  = xk[2];
    
	x4_kp1  = c4[0]*xk[0] +
              c4[1]*xk[1] + 
              c4[2]*xk[2] +
              c4[3]*xk[3] + 
              c4[4]*xk[4] +
              c4[5]*xk[5] + 
              c4[6]*u;

	x5_kp1  = xk[4];
    
	x6_kp1  = c5[0]*xk[0] +
              c5[1]*xk[1] + 
              c5[2]*xk[2] +
              c5[3]*xk[3] + 
              c5[4]*xk[4] +
              c5[5]*xk[5] + 
              c5[6]*xk[6] +
              c5[7]*xk[7] + 
              c5[8]*u;

	x7_kp1  = xk[6];
    
	x8_kp1  = c6[0]*xk[0] +
              c6[1]*xk[1] + 
              c6[2]*xk[2] +
              c6[3]*xk[3] + 
              c6[4]*xk[4] +
              c6[5]*xk[5] + 
              c6[6]*xk[6] +
              c6[7]*xk[7] + 
              c6[8]*xk[8] +
              c6[9]*xk[9] + 
              c6[10]*u;

	x9_kp1  = xk[8];
    
	x10_kp1  = c7[0]*xk[0] +
               c7[1]*xk[1] + 
               c7[2]*xk[2] +
               c7[3]*xk[3] + 
               c7[4]*xk[4] +
               c7[5]*xk[5] + 
               c7[6]*xk[6] +
               c7[7]*xk[7] + 
               c7[8]*xk[8] +
               c7[9]*xk[9] + 
               c7[10]*xk[10] +
               c7[11]*xk[11] + 
               c7[12]*u;

	x11_kp1  = xk[10];


	//permutations
	xk[0] = x0_kp1;
	xk[1] = x1_kp1;
	xk[2] = x2_kp1;
	xk[3] = x3_kp1;
	xk[4] = x4_kp1;
	xk[5] = x5_kp1;
	xk[6] = x6_kp1;
	xk[7] = x7_kp1;
	xk[8] = x8_kp1;
	xk[9] = x9_kp1;
	xk[10] = x10_kp1;
	xk[11] = x11_kp1;


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
  __PROMISE__ xk[12] = {};

  

 for (int i=0; i<=99; i++){
   u = rand2() % (1 + 1 + 1) - 1;
   z= dSS(u,xk);  

     

  }  

  PROMISE_CHECK_VAR(z);

  cout<<z<<endl;
    
    return 0;
    
}
