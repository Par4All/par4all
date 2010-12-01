#include <stdio.h>
#include <complex.h>

int main (void){
  complex double z = 42.0 + 42.0*I;

  printf("z = %f + %fI\n", creal(z), cimag(z));

  z = z * z;

  printf("z * z = %f + %fI\n", creal(z), cimag(z));

  return 0;
}
