#include<stdio.h>
int main()
{
  int *q;
  int *p;
  int i=0, j=1;
  q = &i;
  p = i>0 ? &j : q;
  printf("address of p =%p\n",p);
  printf("address of q =%p\n",q);
  return 0;
}
