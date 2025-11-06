#include <iostream>



void dSS( __PROMISE__ y[7],  __PROMISE__ u[5],  __PROMISE__ xk[4])
{
	 __PROMISE__ x0_kp1;
     __PROMISE__ x1_kp1;
     __PROMISE__ x2_kp1;
     __PROMISE__ x3_kp1;


	// intermediate variable(s)
    __PROMISE__ c1[6];
    c1[0] = 0x1.ff5e47890c8b0p+9;
    c1[1] = 0x1.7c21efdc21b48p-1;
    c1[2] = 0x1.231c60afeb26cp+1;
    c1[3] = 0x1.4370ede6e9ff9p+0;
    c1[4] = -0x1.7c21efdc21b48p-1;
    c1[5] = -0x1.231c60afeb26cp+1;

    
    __PROMISE__ c2[6];
    c2[0] = 0x1.7c21efdc21b48p-1;
    c2[1] = 0x1.ffc73dd15a6e6p+9;
    c2[2] = -0x1.8ecf0df481e0cp+0;
    c2[3] = -0x1.7c21efdc21b48p-1;
    c2[4] = 0x1.c611752c8cf27p-2;
    c2[5] = 0x1.8ecf0df481e0cp+0;
    
    
    __PROMISE__ c3[6];
    c3[0] = 0x1.231c60afeb26cp+1;
    c3[1] = -0x1.8ecf0df481e0cp+0;
    c3[2] = 0x1.fa366f1b68439p+9;
    c3[3] = -0x1.231c60afeb26cp+1;
    c3[4] = 0x1.8ecf0df481e0cp+0;
    c3[5] = 0x1.72643925ef1abp+3;
    
    
    __PROMISE__ c4[7];
    c4[0] = 0x1.0000000000000p+10;
    c4[1] = 0x1.a14e7a5d9e9b8p-1;
    c4[2] = -0x1.f78ca5c1afff1p-2;
    c4[3] = -0x1.146c81b5823c0p+1;
    c4[4] = -0x1.a14e7a5d9e9b8p-1;
    c4[5] = 0x1.f78ca5c1afff1p-2;
    c4[6] = 0x1.146c81b5823c0p+1;
    
    
    __PROMISE__ c5[6];
    c5[0] = 0x1.ff5e47890c8b0p+9;
    c5[1] = 0x1.7c21efdc21b48p-1;
    c5[2] = 0x1.231c60afeb26cp+1;
    c5[3] = 0x1.4370ede6e9ff9p+0;
    c5[4] = -0x1.7c21efdc21b48p-1;
    c5[5] = -0x1.231c60afeb26cp+1;
    
    __PROMISE__ c6[6];
    c6[0] = 0x1.7c21efdc21b48p-1;
    c6[1] = 0x1.ffc73dd15a6e6p+9;
    c6[2] = -0x1.8ecf0df481e0cp+0;
    c6[3] = -0x1.7c21efdc21b48p-1;
    c6[4] = 0x1.c611752c8cf27p-2;
    c6[5] = 0x1.8ecf0df481e0cp+0;
    
    
    __PROMISE__ c7[6];
    c7[0] = 0x1.231c60afeb26cp+1;
    c7[1] = -0x1.8ecf0df481e0cp+0;
    c7[2] = 0x1.fa366f1b68439p+9;
    c7[3] = -0x1.231c60afeb26cp+1;
    c7[4] = 0x1.8ecf0df481e0cp+0;
    c7[5] = 0x1.72643925ef1abp+3;
    
    
    
    __PROMISE__ c8[9];
    c8[0] = 0x1.e63422711ae64p-1;
    c8[1] = -0x1.c34da2bd832cap-11;
    c8[2] = 0x1.d296649788b11p-5;
    c8[3] = -0x1.0eae3e99fdbefp-9;
    c8[4] = 0x1.57b0e8e94b10ep-4;
    c8[5] = -0x1.ea850ffde6435p-17;
    c8[6] = -0x1.a3392964fbc36p-11;
    c8[7] = 0x1.f99183de06740p-12;
    c8[8] = 0x1.1279f79e6d38bp-9;
    
    __PROMISE__ c9[9];
    c9[0] = 0x1.3f59a6c2898c8p-9;
    c9[1] = 0x1.f24cd6ab1adbep-1;
    c9[2] = -0x1.9acc5bd7171e9p-5;
    c9[3] = 0x1.21d726db12db5p-9;
    c9[4] = 0x1.20e07be578da6p-4;
    c9[5] = -0x1.41bc052ca5b75p-26;
    c9[6] = 0x1.445cdcae37721p-10;
    c9[7] = -0x1.7d63770d231a5p-11;
    c9[8] = -0x1.2530c287a6874p-9;
    
    
    __PROMISE__ c10[9];
    c10[0] = 0x1.28d5c101f1d24p-15;
    c10[1] = 0x1.7c1ad030a230ap-11;
    c10[2] = 0x1.ffc2a877d64d3p-1;
    c10[3] = -0x1.1746dbfa8dbf4p-11;
    c10[4] = -0x1.ef34efec02b38p-15;
    c10[5] = 0x1.a76391bc769d4p-27;
    c10[6] = -0x1.7d4cef41af85cp-11;
    c10[7] = 0x1.c7aaf38717382p-12;
    c10[8] = 0x1.91c78a9c20fd2p-10;
    
    __PROMISE__ c11[9];
    c11[0] = 0x1.1f646f6d987aep-4;
    c11[1] = -0x1.31165e425b8f7p-9;
    c11[2] = -0x1.153f784d00ef4p-4;
    c11[3] = 0x1.fa2a91b4f1a87p-1;
    c11[4] = -0x1.dfb16c0d2438cp-4;
    c11[5] = 0x1.9a85494751478p-16;
    c11[6] = -0x1.24df2773a8594p-9;
    c11[7] = 0x1.90fc3c8accd55p-10;
    c11[8] = 0x1.7443085561e94p-7;

	//output(s)
	y[0]  = c1[0]*xk[1] +
            c1[1]*xk[2] + 
            c1[2]*xk[3] +
            c1[3]*u[2] + 
            c1[4]*u[3] +
            c1[5]*u[4];


	y[1]  = c2[0]*xk[1] +
            c2[1]*xk[2] + 
            c2[2]*xk[3] +
            c2[3]*u[2] + 
            c2[4]*u[3] +
            c2[5]*u[4];


	y[2]  = c3[0]*xk[1] +
            c3[1]*xk[2] + 
            c3[2]*xk[3] +
            c3[3]*u[2] + 
            c3[4]*u[3] +
            c3[5]*u[4];


	y[3]  = c4[0]*xk[0] +
            c4[1]*xk[1] + 
            c4[2]*xk[2] +
            c4[3]*xk[3] + 
            c4[4]*u[2] +
            c4[5]*u[3] + 
            c4[6]*u[4];


	y[4]  = c5[0]*xk[1] +
            c5[1]*xk[2] + 
            c5[2]*xk[3] +
            c5[3]*u[2] + 
            c5[4]*u[3] +
            c5[5]*u[4];


	y[5]  = c6[0]*xk[1] +
            c6[1]*xk[2] + 
            c6[2]*xk[3] +
            c6[3]*u[2] + 
            c6[4]*u[3] +
            c6[5]*u[4];


	y[6]  = c7[0]*xk[1] +
            c7[1]*xk[2] + 
            c7[2]*xk[3] +
            c7[3]*u[2] + 
            c7[4]*u[3] +
            c7[5]*u[4];


	//states
	x0_kp1  = c8[0]*xk[0] +
              c8[1]*xk[1] + 
              c8[2]*xk[2] +
              c8[3]*xk[3] + 
              c8[4]*u[0] +
              c8[5]*u[1] + 
              c8[6]*u[2] +
              c8[7]*u[3] + 
              c8[8]*u[4];


	x1_kp1  = c9[0]*xk[0] +
              c9[1]*xk[1] + 
              c9[2]*xk[2] +
              c9[3]*xk[3] + 
              c9[4]*u[0] +
              c9[5]*u[1] + 
              c9[6]*u[2] +
              c9[7]*u[3] + 
              c9[8]*u[4];



	x2_kp1  = c10[0]*xk[0] +
              c10[1]*xk[1] + 
              c10[2]*xk[2] +
              c10[3]*xk[3] + 
              c10[4]*u[0] +
              c10[5]*u[1] + 
              c10[6]*u[2] +
              c10[7]*u[3] + 
              c10[8]*u[4];


	x3_kp1  = c11[0]*xk[0] +
              c11[1]*xk[1] + 
              c11[2]*xk[2] +
              c11[3]*xk[3] + 
              c11[4]*u[0] +
              c11[5]*u[1] + 
              c11[6]*u[2] +
              c11[7]*u[3] + 
              c11[8]*u[4];


	//permutations
	xk[0] = x0_kp1;
	xk[1] = x1_kp1;
	xk[2] = x2_kp1;
	xk[3] = x3_kp1;



}

int seed = 1234;
int rand2(){
    seed = (16807 * seed) % 2147483647;
    return 21878754 * (seed / 2147483647);
}

int main() {
  __PROMISE__ z[7]={};  
  __PROMISE__ u[5] ;
  __PROMISE__ xk[4] = {};


 for (int i=0; i<=99; i++){
  u[0] = rand2() % (1 + 1 + 1) - 1;
  u[1] = rand2() % (1 + 1 + 1) - 1;
  u[2] = rand2() % (1 + 1 + 1) - 1;
  u[3] = rand2() % (1 + 1 + 1) - 1;
  u[4] = rand2() % (1 + 1 + 1) - 1;
  dSS(z,u,xk);  

  }  

  PROMISE_CHECK_ARRAY(z,7);
  for(int i=0; i<7; i++)
    std::cout<<z[i]<<std::endl;

    return 0;

}
