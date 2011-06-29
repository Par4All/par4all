/* Check intermediate returns */

#include <stdio.h>

void for_loop08()
{
    int i;
    for(i=0;i!=5;i++) {
      if (i == 3) {
	printf("i=%d\n",i);
	return;
      }
    }
    printf("Exit with %d\n",i);
}

main()
{
  for_loop08();
  return 0;
}
