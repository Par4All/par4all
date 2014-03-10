// use-def chains with if/else
// no dependence between if and else case have to be done

#include <stdlib.h>

int if01a()
{
  int r;

  if (rand())
    r = 1;
  else
    r = 0;

  return r;
}
