// use-def elimination should have no effects

#include <stdlib.h>

int use_def_elim12a_graph()
{
  int r;

  if (rand())
    r = 1;
  else
    r = 0;

  return r;
}
