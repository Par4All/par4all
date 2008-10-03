/* Check the return type of the function? No, it's implictly int
   according to C standard. */

#include <complex.h>

typedef complex foo;

void generate03()
{
  foo i;
  double x = 1.;
  foo y;

  // use an undeclared function without source code, which returns
  // implictly an inta typedef type
  y = func(i, &x);
}
