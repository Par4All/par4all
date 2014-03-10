// use-def elimination should have no effects
// same than use_def_elim13b but with cumulated effects

#include <stdlib.h>

int use_def_elim13c()
{
  int r=0;

  if (rand())
    r = 1;
  if (rand())
    r = 0;
  
  return r;
}
