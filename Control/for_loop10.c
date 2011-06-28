/* for to do loop conversion */

#include <stdio.h>

int for_loop10()
{
  int i, j = 0;
  for(i=0;i!=5;i++) {
    if (i == 3) {
      printf("i=%d\n",i);
    }
    j++;
  }
  printf("Exit with j=%d\n",j);
  return i;
}

main()
{
  (void) for_loop10();
  return 0;
}
