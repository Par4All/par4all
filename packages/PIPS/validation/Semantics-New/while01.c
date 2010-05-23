/* Make sure the side effect of the condition is taken into account
   when a while loop is not entered. */

#include <stdio.h>

main()
{
  int i = -1;
  int a[20];

  while(i++>0) {
    a[i] = i;
  }
  printf("%d\n", i);
}
