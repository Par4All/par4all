#include<stdlib.h>
#include<stdio.h>
typedef int * pointer;

// should catch that returned value is malloc'ed there
pointer alloc_pointer(int v)
{
  pointer p;
  if(p!=NULL)
     p = malloc(sizeof(int));
  else
     p = malloc(2*sizeof(int));
  *p = v;
  return p;
}

int main(void)
{
  pointer p1, p2;

  // could differentiate allocs based on call path?
  p1 = alloc_pointer(13);
  p2 = p1;
 

  return;
}
