/* Check the handling of comma expression */
#include<stdio.h>
int main ()
{
  int i = 0, j = 1, k = 2, l = 3;
  _Bool b;
  int * p, *q, * r, * s, *t;
  p = &k;
  r = &i;
  q = &j;
  s = &l;
  /* printf("address of p=%p\n ",p); */
/*   printf("address of r=%p\n",r); */
/*   printf("address of q=%p\n",q); */
/*   printf("address of s=%p\n",s); */

  b = (p=q) && (r=s);
 /*  printf("address of p=%p\n",p); */
/*   printf("address of r=%p\n",r); */
  if((p=q) && (r=s))
    *p++;
  else if((p=q) || (r=s))
    *r++;

  return 0;

}
