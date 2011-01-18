#include <stdio.h>
#include <complex.h>

int main (void){
  _Complex double z;

  printf("z = %f + %fI\n", creal(z),cimag(z));

  return 0;
}
