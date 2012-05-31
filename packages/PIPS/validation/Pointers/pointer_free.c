#include<stdlib.h>
typedef int * pointer;

// should catch that returned value is malloc'ed there
pointer alloc_pointer(int v)
{
  pointer p = malloc(sizeof(int));
  *p = v;
  return p;
}

void pointer_free(pointer p)
{
  free(p);
  return;
}




int main(void)
{
  pointer p1, p2;

  // could differentiate allocs based on call path?
  p1 = alloc_pointer(13);
  p2 = p1;
  pointer_free(p1);

  return;
}
