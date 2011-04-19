/* Check the handling of comma expression */
#include<stdio.h>
int main ()
{
  int i = 0, j = 1, k = 2, l = 3;
  int * p, *q, * r, *s;

  r = &i;
  q = &j;
  /* here the comma operator defines a sequence point.
     we treat this instructions asfollow :
     p = r;
     q;
     s = p;
  */
  p = r, q, s = p ;
  return 0;
}
