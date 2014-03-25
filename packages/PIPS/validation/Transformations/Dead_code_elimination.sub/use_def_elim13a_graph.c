// use-def elimination should have no effects
// same than use_def_elim12
// but explicit separate if else in 2 if

#include <stdlib.h>

int use_def_elim13a_graph()
{
  int r;

  if (rand())
    r = 1;
  if (rand())
    r = 0;
  
  return r;
}
