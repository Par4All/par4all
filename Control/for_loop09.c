/* Check intermediate returns */

#include <stdio.h>

int for_loop09()
{
  int i, j = 0;
  for(i=0;i!=5;i++) {
    if (i == 3) {
      printf("%d",i);
      return i;
    }
    j++;
  }
  printf("Exit with %d",j);
  return i;
}
