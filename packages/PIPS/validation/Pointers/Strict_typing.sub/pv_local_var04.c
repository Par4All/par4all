/* to test the non-elimination of static variables */

#include <stdio.h>

int a = 0;
int b = 1;

int main()
{
  int * pp = NULL; // FI: to manage an empty points-to set
  if (a>1)
    {
      static int *p = &a;
      static int *q;
      q = &b;
      a = a -1;
    }
  else
    {
      static int *r = &a;
      static int *s;
      s = &b;
      a = a +2;
    }

  return(0);
}
