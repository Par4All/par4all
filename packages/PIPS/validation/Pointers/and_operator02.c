/* Check the handling of comma expression
 *
 * And the side-effects in expressions and conditions...
 */
#include<stdio.h>
int main ()
{
  int i[10], j[20], k[30], l[40];
  _Bool b;
  int * p, *q, * r, * s, *t;
  p = &k[0];
  r = &i[0];
  q = &j[0];
  s = &l[0];
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
