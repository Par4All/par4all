#include <stdio.h>
#include <math.h>
int main()
{
  long double x = 90.0l;
  long double res=acosl(x);
  printf ("acosl(1.0l)=%Lf",res);
  return 0;
}
