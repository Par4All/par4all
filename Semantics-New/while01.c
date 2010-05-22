/* Make sure the side effect of the condition is taken into account
   when a while loop is not entered. */

#include <stdio.h>

main()
{
  int i = -1;

  while(i++>0) {
    ;
  }
  printf("%d\n", i);
}
