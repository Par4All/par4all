/* frexp example : Get significand and exponent of 'x'
Breaks the floating point number 'x' into its binary significand (a floating point value between 0.5(included) and 1.0(excluded)) and an integral exponent for 2, such that:

x = significand * 2 exponent*/
#include <stdio.h>
#include <math.h>

int main ()
{
  double x, significand;
  int exponent;

  x = 8.0;
  significand = frexp (x , &exponent);
  printf ("%lf * 2^%d = %f\n", significand, exponent, x);
  return 0;
}
