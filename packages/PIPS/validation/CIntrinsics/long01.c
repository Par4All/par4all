#include <stdio.h>
#include <complex.h>

int main()
{
  int a = 11;
  long b = 1000000000;
  long int c = 200000;
  long long d = 20480744900000585;
  long long int e = 3669993212;
  double f = 2560.000;
  long double g = 90.500l;
  float complex h; // = 2.1 + 7.7*I;
  double complex z;  //= 42.1550 + 42.2000*I;
  long double complex y; // = 8.6523l + 98.23541l*I;

  //printf("float complex h= %f + %fI, double complex z= %f + %fI, long double complex y= %Lf + %LfI \n", crealf(h), cimagf(h), creal(z), cimag(z), creall(y), cimagl(y));
  printf ("int a=%d, long b=%ld, long int c=%ld, long long d=%lld, long long int e=%lld, double f=%f, long double g=%Lf\n",a,b,c,d,e,f,g);
  return 0;
}
