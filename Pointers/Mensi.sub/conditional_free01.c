/* To illustrate the interprocedural issues with free() */

#include <stdlib.h>
#include <stdbool.h>

void conditional_free01(int *p, bool c1, bool c2, bool c3) {
  if(c1) p = (int *) malloc(sizeof(int));
  if(c2) free(p);
  if(c3) p = (int *) malloc(sizeof(int));
  return;
}

int main()
{
  int *p = (int *) malloc(sizeof(int));
  conditional_free01(p, false, true, false);
  return 0;
}
