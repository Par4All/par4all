#include <stdio.h>
#include <complex.h>

int main (void){
  _Complex double z;

  printf("z = %f + %fI\n", __real__(z),__imag__(z));

  return 0;
}
