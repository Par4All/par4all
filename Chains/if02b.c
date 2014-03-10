// use-def chains with 2 if
// and declaration with init

#include <stdlib.h>

int if02b()
{
  int r=0;

  if (rand())
    r = 1;
  if (rand())
    r = 0;
  
  return r;
}
