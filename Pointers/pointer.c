/* FI: I do not know who designed this test case
 *
 * I believe a main is useful. And it should be analyzed interprocedurally
 */

#include <stdlib.h>
#include <stdio.h>

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

int main()
{
  pointer p1 = alloc_pointer(1);
  pointer p2 = alloc_pointer(0);
  pointer p3 = alloc_pointer(0);
  pointer_set(p2, 2);
  pointer_add(p3, p1, p2);
  pointer_free(p1);
  pointer_free(p2);
  printf("*p3=%d\n", pointer_get(p3));
  return 0;
}
