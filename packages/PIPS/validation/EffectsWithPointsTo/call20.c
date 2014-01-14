/* A precise points-to analysis of this program requires
 * context-sensitivity.
 */

#include <stdio.h>
#include <stdlib.h>

typedef int * pointer;

// should catch that returned value is malloc'ed there
pointer pointer_alloc(int v)
{
  pointer p = malloc(sizeof(int));
  *p = v;
  return p;
}

void pointer_free(pointer p)
{
  free(p);
}

// no pointer assignment, no change in points-to
void pointer_set(pointer p, int v)
{
  *p = v;
}

// no pointer assignment, no change in points-to
void pointer_add(pointer q1, const pointer q2, const pointer q3)
{
  *q1 = (*q2) + (*q3);
}

// no pointer assignment, no change in points-to
int pointer_get(const pointer p)
{
  return *p;
}

int main(void)
{
  pointer p1, p2, p3;

  // could differentiate allocs based on call path?
  p1 = pointer_alloc(13);
  p2 = pointer_alloc(17);
  p3 = pointer_alloc(19);

  // no pointer assigned! can keep all points-to
  pointer_add(p1, p2, p3);

  fprintf(stdout, "p1=%d p2=%d p3=%d\n", // 36 17 19
          pointer_get(p1), pointer_get(p2), pointer_get(p3));

  // no pointer assigned! can keep all points-to
  pointer_set(p3, 23);

  // no pointer assigned! can keep all points-to
  pointer_add(p3, p2, p1);

  fprintf(stdout, "p1=%d p2=%d p3=%d\n", // 36 17 53
          pointer_get(p1), pointer_get(p2), pointer_get(p3));

  pointer_free(p1);
  pointer_free(p2);
  pointer_free(p3);
}
