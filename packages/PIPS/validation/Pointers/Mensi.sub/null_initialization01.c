#include<stdio.h>

int foo(int *q)
{
  int i = 0, *p;
  p = q;
  if(p==NULL)
    printf("i = %d", i);
  return i;
}
  
