#include <stdio.h>

int main (void){
  _Complex double z;

  z = 42.0 + 42.0fi;

  printf("z = %f + %fI\n", __real__(z), __imag__(z));

  return 0;
}
