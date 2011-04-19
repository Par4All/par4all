/* Check the handling of comma expression */
#include<stdio.h>
int main ()
{
  int i = 0, j = 1, k = 2, l = 3;
  int * p, *q, * r, * s, *t;

  r = &i;
  q = &j;
  s = &k;
  /* we treat this instruction as:
     r = &j;
     r = s;
     p = r;
  */
  p = (r = &j, r = s);
  return 0;
}
