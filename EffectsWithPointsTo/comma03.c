/* Check the handling of comma expression */
#include<stdio.h>
int main ()
{
  int i = 0, j = 1, k = 2, l = 3;
  int * p, *q, * r, * s, *t;

  r = &i;
  q = &j;
  s = &k;
  /* we trat this instruction like :
     s = r;
     q;
     s;
     p = s;
  */
  p = (s = r, q, s);
  return 0;
}
