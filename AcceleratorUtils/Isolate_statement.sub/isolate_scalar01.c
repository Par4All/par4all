#include <stdio.h>
void lonesome_cowboy()
{
  int i, j;
  i = 42;
  j = 314;
 isolate:
  j = i;
  i = 2718;
  printf("i =%d, j =%d", i, j);
}

main()
{
    lonesome_cowboy();
    return 0;
}
