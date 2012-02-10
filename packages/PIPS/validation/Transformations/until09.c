// Check that side effects prevent simplification

// Although the simplification could be implemented by improvoing the transformation

#include <stdio.h>

void until09()
{
  int i = 1;
  do {
    i++;
  } while(i++<0);
  printf("%d", i);
}
