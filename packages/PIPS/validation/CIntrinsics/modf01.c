/* modf example : Breaks 'param' into two parts: the integer part (stored in the object pointed by intpart) and the fractional part (returned by the function). */
#include <stdio.h>
#include <math.h>

int main ()
{
  double param, fractpart, intpart;

  param = 3.14159265;
  fractpart = modf (param , &intpart);
  printf ("%lf = %lf + %lf \n", param, intpart, fractpart);
  return 0;
}
