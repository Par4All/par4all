#include <stdio.h>
#include <stdlib.h>

#include "pointer.c"

int main(void)
{
  int i1 = 13, i2 = 17, i3 = 19;
  pointer p1, p2, p3;

  p1 = &i1;
  p2 = &i2;
  p3 = &i3;

  // no pointer assigned! can keep all points-to
  pointer_add(p1, p2, p3);
  fprintf(stdout, "p1=%d p2=%d p3=%d\n", // 36 17 19
          // no pointer assigned
          pointer_get(p1), pointer_get(p2), pointer_get(p3));

  // no pointer assigned! can keep all points-to
  pointer_set(p3, 23);

  // no pointer assigned! can keep all points-to
  pointer_add(p3, p2, p1);

  fprintf(stdout, "p1=%d p2=%d p3=%d\n", // 36 17 53
          // no pointer assigned
          pointer_get(p1), pointer_get(p2), pointer_get(p3));
}
