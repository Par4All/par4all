/* Make sure that C formal parameters are substituted when possible */

#include <stdio.h>

int duck(int size)
{
  printf("size=%d\n", size);
  return size;
}

int main()
{
  int size = 2, t;
  t = duck(size);
  printf("%d\n", t);
  return 0;
}
