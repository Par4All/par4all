/* Check interface with effects_with_points_to */

#include <stdio.h>

void fprintf02(int *p)
{
  printf("%d", *p);
  return;
}
