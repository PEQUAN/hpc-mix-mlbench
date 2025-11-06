#include <iostream>
#include <fstream>


using namespace std;


void calcul(int n, int nbiter, __PR_1__ u[101]){
    
     ofstream myfile;
     myfile.open("half_result.txt");
     __PROMISE__ v[101];
     __PROMISE__ c1, c2;
     
     
     c1=3./16.;
     c2=5./8.;
     
     //other possibility (not used here):
     //c1=1./6.;
     //c2=2./3.;
     
     
    for (int i=0; i<=n; i++)
       {
	 v[i]=u[i];
         //myfile <<"u("<< i <<")=" << strp(u[i])<<endl;
       }
     for (int l=1; l<=nbiter; l++){
         myfile <<"--------------------------"<<endl;
         myfile <<"iteration="<<l<<endl;
         myfile <<"--------------------------"<<endl;
        for (int j=1; j <=n-1; j++){
	      u[j] = c1*v[j-1]+c2*v[j]+c1*v[j+1];
          myfile<<u[j]<<endl;
           }
    
     for (int k=0; k <=n; k++)
     v[k]= u[k];
      }
         myfile.close();

     
}

void init ( __PR_1__ u[101],int n){
    double pi=3.141592653589793238462643383795;
    for (int i=0; i<=n; i++){
       u[i]=sin(i*pi/n);
      /*  
    if ((i % 2) ==0)
        u[i]= i;
    else
        u[i]=i/16.;
      */
    }

}

int main ()
{
     int n = 100;
     int nbiter = 100;
     __PR_1__ u[101];

     init(u, n);
     calcul(n,nbiter,u);
     PROMISE_CHECK_ARRAY(u,n);

    return 0;
} 
