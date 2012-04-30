/* Check the dereferencing of a null pointer */

#include <stdio.h>

int main()
{
  int * p, i;
  p = NULL;

  if(i) p = &i;

  *p = 1;

  return 0;
}
