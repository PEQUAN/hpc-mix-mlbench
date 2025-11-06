#include <iostream>



using namespace std;
__PROMISE__ dSS(__PROMISE__ u, __PROMISE__ xk[10])
{
	__PROMISE__ x0_kp1, x1_kp1, x2_kp1, x3_kp1, x4_kp1, x5_kp1, x6_kp1, x7_kp1, x8_kp1, x9_kp1;
	__PROMISE__ y;

	// intermediate variable(s)
  __PROMISE__ c1[11];
  c1[0] = -0x1.84b342095a792p-6;
  c1[1] = 0x1.a03aa3ce3fa7cp-6;
  c1[2] = 0x1.3d2a17a781b5cp-10;
  c1[3] = -0x1.a4314c09594cdp-20;
  c1[4] = 0x1.476af16f622d2p-11;
  c1[5] = 0x1.79127b1ff0e15p-7;
  c1[6] = 0x1.ac21fd344d597p-12;
  c1[7] = -0x1.278c03c8c7e59p-9;
  c1[8] = 0x1.4060ab24497aap-13;
  c1[9] = 0x1.4626a923e5149p-17;
  c1[10] = -0x1.b652a2fb7a7dfp-3; 

  
  
  __PROMISE__ c2[11];
  c2[0] = 0x1.a39419b15d3b8p-1;
  c2[1] = 0x1.1feb53ef06fe1p-2;
  c2[2] = -0x1.0fc47290d2cffp-5;
  c2[3] = -0x1.ba277f245c67ap-6;
  c2[4] = -0x1.51b3bea9e450fp-3;
  c2[5] = 0x1.0dfedaffa00a0p-3;
  c2[6] = 0x1.5af67e3aad17ep-7;
  c2[7] = -0x1.13ca4673454ccp-4;
  c2[8] = 0x1.cacd4a561f44bp-10;
  c2[9] = 0x1.11b090602251cp-14;
  c2[10] = -0x1.2fac0b974e944p+1;
  
  
  

  __PROMISE__ c3[11];
  c3[0] = -0x1.1feb53ef06e7cp-2;
  c3[1] = -0x1.ed413c063dd55p-2;
  c3[2] = -0x1.5585aa7862115p-3;
  c3[3] = 0x1.62790fd5a85bap-4;
  c3[4] = -0x1.14a1569134b84p-1;
  c3[5] = 0x1.2ce927c561575p-3;
  c3[6] = 0x1.2cfcce255d222p-6;
  c3[7] = -0x1.efeb3e3003917p-4;
  c3[8] = 0x1.fd27783b23e02p-10;
  c3[9] = 0x1.6607a5d2f151bp-16;
  c3[10] = -0x1.452dcff921e12p+1;
  
  
  __PROMISE__ c4[11];
  c4[0] = 0x1.0fc47290d1914p-5;
  c4[1] = -0x1.5585aa786115bp-3;
  c4[2] = 0x1.f32b0d8471cd1p-1;
  c4[3] = 0x1.15d3cdcc5b962p-6;
  c4[4] = -0x1.74e60a8d02eaap-4;
  c4[5] = 0x1.38d9a7421e1fbp-4;
  c4[6] = 0x1.b7fb51d86492cp-9;
  c4[7] = -0x1.48b629eb89522p-6;
  c4[8] = 0x1.ba91dfdc82f0fp-11;
  c4[9] = 0x1.7d83a58f598b5p-15;
  c4[10] = -0x1.ef91c4f5ba646p-4;
  
  __PROMISE__ c5[11];
  c5[0] = 0x1.ba277f20d0ccdp-6;
  c5[1] = -0x1.62790fd2b43eap-4;
  c5[2] = -0x1.15d3cdca5b566p-6;
  c5[3] = 0x1.eb90cadf1391ep-1;
  c5[4] = 0x1.02e875f0dfa7ap-2;
  c5[5] = 0x1.e7f12ea1f13b3p-5;
  c5[6] = 0x1.b185419b479fap-10;
  c5[7] = -0x1.29b253023632ap-7;
  c5[8] = 0x1.3cfd020b87693p-11;
  c5[9] = 0x1.4269cebbfa207p-15;
  c5[10] = -0x1.484682a655228p-13;
  
  
  __PROMISE__ c6[11];
  c6[0] = 0x1.51b3bea9f73afp-3;
  c6[1] = -0x1.14a15691441d1p-1;
  c6[2] = -0x1.74e60a8d165dap-4;
  c6[3] = -0x1.02e875f06237ap-2;
  c6[4] = 0x1.345679c86cacep-1;
  c6[5] = 0x1.8e1e9e06fd9cdp-2;
  c6[6] = 0x1.78ade6006a193p-7;
  c6[7] = -0x1.06fb3a33f8d79p-4;
  c6[8] = 0x1.02918cb945577p-8;
  c6[9] = 0x1.01567bc10ee5fp-12;
  c6[10] = -0x1.ff97193ddcf81p-5;
  
  
  __PROMISE__ c7[11];
  c7[0] = 0x1.0dfedaff9cea9p-3;
  c7[1] = -0x1.2ce927c55ce40p-3;
  c7[2] = -0x1.38d9a7421eb1fp-4;
  c7[3] = 0x1.e7f12ea636c40p-5;
  c7[4] = -0x1.8e1e9e06ea613p-2;
  c7[5] = 0x1.dd94d64fabf58p-2;
  c7[6] = -0x1.fc5edf39b6acep-5;
  c7[7] = 0x1.b07eae8406203p-2;
  c7[8] = -0x1.bd1860ec880acp-11;
  c7[9] = 0x1.843eed27448f8p-12;
  c7[10] = 0x1.26967030f381fp+0;
  
  
  
  __PROMISE__ c8[11];
  c8[0] = 0x1.5af67e3a9da07p-7;
  c8[1] = -0x1.2cfcce25521a3p-6;
  c8[2] = -0x1.b7fb51d8594f7p-9;
  c8[3] = 0x1.b185419f83442p-10;
  c8[4] = -0x1.78ade6004e758p-7;
  c8[5] = -0x1.fc5edf39ac718p-5;
  c8[6] = 0x1.f76af2c730050p-1;
  c8[7] = 0x1.01aa9ad9a5e12p-3;
  c8[8] = 0x1.fb06199ef8c00p-8;
  c8[9] = 0x1.4f20fedadde40p-11;
  c8[10] = 0x1.4e7a8dd0d15ebp-5;
  
  
  
  __PROMISE__ c9[11];
  c9[0] = 0x1.13ca4673469cep-4;
  c9[1] = -0x1.efeb3e30042e8p-4;
  c9[2] = -0x1.48b629eb8a2fdp-6;
  c9[3] = 0x1.29b25304f162ap-7;
  c9[4] = -0x1.06fb3a33ee313p-4;
  c9[5] = -0x1.b07eae8404c8fp-2;
  c9[6] = -0x1.01aa9ad99e0d6p-3;
  c9[7] = -0x1.6f387639a667cp-5;
  c9[8] = 0x1.294a31a70cb4dp-4;
  c9[9] = 0x1.710189b21e83dp-8;
  c9[10] = 0x1.cdcac5e9b80a3p-3;
  
  
  
  __PROMISE__ c10[11];
  c10[0] = -0x1.cacd4a561e104p-10;
  c10[1] = 0x1.fd27783b107a0p-10;
  c10[2] = 0x1.ba91dfdc7457cp-11;
  c10[3] = -0x1.3cfd020e2cce2p-11;
  c10[4] = 0x1.02918cb936a61p-8;
  c10[5] = 0x1.bd1860ec4371dp-11;
  c10[6] = -0x1.fb06199f0c27bp-8;
  c10[7] = 0x1.294a31a70cdeep-4;
  c10[8] = 0x1.f7b1f3b5dacbap-1;
  c10[9] = -0x1.44563d3add3fbp-9;
  c10[10] = -0x1.f4970b68a9362p-7;
  
  
  __PROMISE__ c11[11];
  c11[0] = -0x1.11b0905e1bff9p-14;
  c11[1] = 0x1.6607a5dac7642p-16;
  c11[2] = 0x1.7d83a58e8c152p-15;
  c11[3] = -0x1.4269cebcd5e4fp-15;
  c11[4] = 0x1.01567bc04ea2bp-12;
  c11[5] = -0x1.843eed27a4ad6p-12;
  c11[6] = -0x1.4f20feda527d1p-11;
  c11[7] = 0x1.710189b121a4fp-8;
  c11[8] = -0x1.44563d3a15e3fp-9;
  c11[9] = 0x1.e239e73049a5cp-1;
  c11[10] = -0x1.fd9c6845c0c37p-11;
  
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
		 c1[10]*u;


	//states
	x0_kp1  = c2[0]*xk[0] +
			  c2[1]*xk[1] +
			  c2[2]*xk[2] +
			  c2[3]*xk[3] +
			  c2[4]*xk[4] +
			  c2[5]*xk[5] +
			  c2[6]*xk[6] +
			  c2[7]*xk[7] +
			  c2[8]*xk[8] +
			  c2[9]*xk[9] +
			  c2[10]*u;



	x1_kp1  = c3[0]*xk[0] +
			  c3[1]*xk[1] +
			  c3[2]*xk[2] +
			  c3[3]*xk[3] +
			  c3[4]*xk[4] +
			  c3[5]*xk[5] +
			  c3[6]*xk[6] +
			  c3[7]*xk[7] +
			  c3[8]*xk[8] +
			  c3[9]*xk[9] +
			  c3[10]*u;



	x2_kp1  = c4[0]*xk[0] +
			  c4[1]*xk[1] +
			  c4[2]*xk[2] +
			  c4[3]*xk[3] +
			  c4[4]*xk[4] +
			  c4[5]*xk[5] +
			  c4[6]*xk[6] +
			  c4[7]*xk[7] +
			  c4[8]*xk[8] +
			  c4[9]*xk[9] +
			  c4[10]*u;


	x3_kp1  = c5[0]*xk[0] +
			  c5[1]*xk[1] +
			  c5[2]*xk[2] +
			  c5[3]*xk[3] +
			  c5[4]*xk[4] +
			  c5[5]*xk[5] +
			  c5[6]*xk[6] +
			  c5[7]*xk[7] +
			  c5[8]*xk[8] +
			  c5[9]*xk[9] +
			  c5[10]*u;


	x4_kp1  = c6[0]*xk[0] +
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


	x5_kp1  = c7[0]*xk[0] +
			  c7[1]*xk[1] +
			  c7[2]*xk[2] +
			  c7[3]*xk[3] +
			  c7[4]*xk[4] +
			  c7[5]*xk[5] +
			  c7[6]*xk[6] +
			  c7[7]*xk[7] +
			  c7[8]*xk[8] +
			  c7[9]*xk[9] +
			  c7[10]*u;


	x6_kp1  = c8[0]*xk[0] +
			  c8[1]*xk[1] +
			  c8[2]*xk[2] +
			  c8[3]*xk[3] +
			  c8[4]*xk[4] +
			  c8[5]*xk[5] +
			  c8[6]*xk[6] +
			  c8[7]*xk[7] +
			  c8[8]*xk[8] +
			  c8[9]*xk[9] +
			  c8[10]*u;


	x7_kp1  = c9[0]*xk[0] +
			  c9[1]*xk[1] +
			  c9[2]*xk[2] +
			  c9[3]*xk[3] +
			  c9[4]*xk[4] +
			  c9[5]*xk[5] +
			  c9[6]*xk[6] +
			  c9[7]*xk[7] +
			  c9[8]*xk[8] +
			  c9[9]*xk[9] +
			  c9[10]*u;



	x8_kp1  = c10[0]*xk[0] +
			  c10[1]*xk[1] +
			  c10[2]*xk[2] +
			  c10[3]*xk[3] +
			  c10[4]*xk[4] +
			  c10[5]*xk[5] +
			  c10[6]*xk[6] +
			  c10[7]*xk[7] +
			  c10[8]*xk[8] +
			  c10[9]*xk[9] +
			  c10[10]*u;


	x9_kp1  = c11[0]*xk[0] +
			  c11[1]*xk[1] +
			  c11[2]*xk[2] +
			  c11[3]*xk[3] +
			  c11[4]*xk[4] +
			  c11[5]*xk[5] +
			  c11[6]*xk[6] +
			  c11[7]*xk[7] +
			  c11[8]*xk[8] +
			  c11[9]*xk[9] + c11[10]*u;


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
	__PROMISE__ xk[10]={};

	for (int i=0; i<=99; i++){
		u = rand2() % (1 + 1 + 1) - 1;
		z = dSS(u,xk);

		PROMISE_CHECK_VAR(z);
	}

	cout << z << endl;
	return 0;

}
