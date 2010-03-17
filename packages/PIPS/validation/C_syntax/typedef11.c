/* typedef scope in preprocessor: here "cookie" is not a typedef */
/* Case used to avoid quick fixes for typedef06.c */

//#include <stdio.h>

typedef int km;

int typedef11(km x)
{
  typedef float km;
  km y;

  // Used to check that x is int and y float as expected
  //printf("%d, %f\n", x, y);
}
