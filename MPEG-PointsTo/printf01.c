// make sure that side effects are taken into account when intrinsics
// are called

#include <stdio.h>

int printf01()
{
  int i = 3;

  printf("%d\n", i++);

  return i;
}
