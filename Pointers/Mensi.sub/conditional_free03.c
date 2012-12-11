/* To illustrate the interprocedural issues with free() */

#include <stdlib.h>
#include <stdbool.h>

void conditional_free03(int *p, bool c1) {
  int *q = p;

  if(c1) free(p);

  return *q;
}

int main()
{
  int *p = (int *) malloc(sizeof(int));
  conditional_free03(p, false);
  return 0;
}
