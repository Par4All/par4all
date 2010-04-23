/* bug seen in Transformations/eval.c: how do we guess effects on
   Fortran parameters? How dowe hanfle C differently from Fortran? */

#include <stdio.h>

int main()
{
  int i;
  i = (-3)%(-2);
  printf("%d\n", i);
  return i;
}
