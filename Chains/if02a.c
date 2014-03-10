// use-def chains with 2 if

#include <stdlib.h>

int if02a()
{
  int r;

  if (rand())
    r = 1;
  if (rand())
    r = 0;
  
  return r;
}
