// use-def elimination should have no effects
// same than use_def_elim13 but declaration with init

#include <stdlib.h>

int use_def_elim13b()
{
  int r=0;

  if (rand())
    r = 1;
  if (rand())
    r = 0;
  
  return r;
}
