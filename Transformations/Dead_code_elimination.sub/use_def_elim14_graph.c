// use-def elimination should have no effects
// same that use_def_elim12
// but with braces to be sure it change nothing for the analysis

#include <stdlib.h>

int use_def_elim14_graph()
{
  int r;

  if (rand())
  {
    r = 1;
    r = r;
  }
  else
  {
    r = 0;
    r = r;
  }

  return r;
}
