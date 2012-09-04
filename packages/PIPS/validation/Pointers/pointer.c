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

// no pointer assignment, no change in points-to
void pointer_set(pointer p, int v)
{
  *p = v;
  return;
}

// no pointer assignment, no change in points-to
void pointer_add(pointer q1, const pointer q2, const pointer q3)
{
  *q1 = (*q2) + (*q3);
  return;
}

// no pointer assignment, no change in points-to
int pointer_get(const pointer p)
{
  int i = *p;
  return i;
}
