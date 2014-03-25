// use-def elimination should have no effects
// code with if with no bug

#include <stdlib.h>

int use_def_elim15()
{
  int r, r1=0, r2=0;

  if (rand())
    r1 = 10;
  else
    r2 = 50;
  
  r= r1+r2;
  return r;
}
