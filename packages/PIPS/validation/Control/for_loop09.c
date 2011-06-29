/* Check intermediate returns */

#include <stdio.h>

int for_loop09()
{
  int i, j = 0;
  for(i=0;i!=5;i++) {
    if (i == 3) {
      printf("i=%d\n",i);
      return i;
    }
    j++;
  }
  printf("Exit with j=%d\n",j);
  return i;
}

main()
{
  for_loop09();
  return 0;
}
