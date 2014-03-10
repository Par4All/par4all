// use-def chains with if/else
// if and else don't work on same variable

#include <stdlib.h>

int if04()
{
  int r, r1=0, r2=0;

  if (rand())
    r1 = 10;
  else
    r2 = 50;
  
  r= r1+r2;
  return r;
}
