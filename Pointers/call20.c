#include <stdio.h>
#include <stdlib.h>

#include "pointer.src/pointer.c"

int main(void)
{
  pointer p1, p2, p3;

  // could differentiate allocs based on call path?
  p1 = alloc_pointer(13);
  p2 = alloc_pointer(17);
  p3 = alloc_pointer(19);

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
  return 0;
}
