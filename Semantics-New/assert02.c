/* Check handling of negation in integer/logical expressions */

// Note the missing parenthesis around c in the if condition
#define assert(c) if(!c) abort();

#include <stdio.h>
#include <stdlib.h>

int assert02(){
  int y, z;

  /*
  z = 1;
  printf("z is positive, !z=%d\n", !z);

  z = 0;
  printf("z is zero, !z=%d\n", !z);

  z = -1;
  printf("z is negative, !z=%d\n", !z);

  if(!z<0)
    abort();
  */

  y = !z;

  z = 1;
  y = !z;

  z = 0;
  y = !z;

  z = -1;
  y = !z;

  return 0;
}

main()
{
  (void) assert02();
}
