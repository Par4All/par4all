/* Check the dereferencing of a null pointer */

#include <stdio.h>

int main()
{
  int * p;
  p = NULL;

  *p = 1;

  return 0;
}
