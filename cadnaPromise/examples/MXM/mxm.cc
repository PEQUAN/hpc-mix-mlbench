# include <cstdlib>
# include <iostream>
# include <fstream>
# include <iomanip>
# include <cmath>
# include <ctime>

typedef __PROMISE__ atype;
typedef __PROMISE__ btype;
typedef __PROMISE__ dtype;
typedef double ieee;

ieee *matgen ( int m, int n, int *seed );

int main ( int argc, char *argv[] )
{
  btype *b;
  dtype *d;
  ieee *tmp;
  int n1;
  int n2;
  int n3;
  int seed;
  n1 = atoi ( argv[1] );
  n2 = atoi ( argv[2] );
  n3 = atoi ( argv[3] );
  
  seed = 42;
  tmp = matgen ( n1, n2, &seed );
  b = new btype[n1*n2];
  d = new dtype[n1*n2];
  for(int i=0; i< n1*n2;i++)
    b[i]=tmp[i];
  delete tmp;
  tmp = matgen ( n2, n3, &seed );
  for(int i=0; i< n2*n3;i++)
    d[i]=tmp[i];
  delete tmp;
  

  atype *a;
  int i;
  int j;
  int k;

  a = new atype[n1*n3];

  for ( j = 0; j < n3; j++ )
  {
    for ( i = 0; i < n1; i++ )
    {
      a[i+j*n1] = 0.0;
    }
  }

  for ( i = 0; i < n1; i++ )
  {
    for ( j = 0; j < n3; j++ )
    {
      for ( k = 0; k < n2; k++ )
      {
        a[i+j*n1] = a[i+j*n1] + b[i+k*n1] * d[k+j*n2];
      }
    }
  }

  PROMISE_CHECK_ARRAY(a, n1*n3);
  delete [] a;


  
  return 0;

 
}
//****************************************************************************80

//****************************************************************************80

ieee *matgen ( int m, int n, int *seed )
{
  ieee *a;
  int i;
  int j;

  a = new ieee[m*n];
  for ( j = 0;j < n; j++ )
  {
    for ( i = 0; i < m; i++ )
    {
      *seed = ( ( 3125 * *seed ) % 65536 );
      a[i+j*m] = ( *seed - 20768.0 ) / 16384.0;
    }
  }

  return a;
}
