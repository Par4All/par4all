/* The printf() is useful to check the scoping used by gcc, but the
   resulting compilation unit makes debugging a pain.*/

#include <stdio.h>

void foo()
{
  printf("call to global function\n");
}
