/* Make sure the side effect of the condition is taken into account
   when a while loop is always entered. */

#include <stdio.h>

main()
{
  int i = 0;
  int a[20];

  while(i++<=10) {
    a[i] = i;
  }
  printf("%d\n", i);
}
