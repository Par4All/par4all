// Check that useless until loops are removed, but that comments are preserved

#include <stdio.h>

void until03()
{
  int i = 1;
  i++;

  /* First loop */
  do {
    int j = 1;
    i += j;
  } while(0);

  /* First loop */
  do {
    /* Second loop body */
    int j = 2;
    i += j;
  } while(0);

  /* Third loop */
  do {
    /* Third loop body */
    int j = 3;
    i += j;
  } while(0);

  printf("%d", i);
}
