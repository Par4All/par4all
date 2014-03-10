// use-def chains with if/else
// and W-W and W-R inside if
// no dependence between if and else case have to be done

#include <stdlib.h>

int if03()
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
