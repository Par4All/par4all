/* Make sure that redeclaration of intrinsic does not break the
   initial declaration in bootstrap */

/*#include <math.h>*/

extern double sqrt(double);

void decl46(double a)
{
  double x;

  x = sqrt(a);
}
