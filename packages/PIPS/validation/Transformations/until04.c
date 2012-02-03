// Check that useless until loops are removed

// Check that labels are preserved...

#include <stdio.h>

void until04()
{
  int i = 1;
 l1:  do {
  l2: i++;
  } while(0);
  printf("%d", i);
  goto l1;
}
