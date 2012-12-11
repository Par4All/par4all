/* To illustrate the interprocedural issues with free() */

#include <stdlib.h>

int ordered_free02(int *fp) {
  int *fq = fp;
  fp = (int *) malloc(sizeof(int));
  free(fp);
  return *fq;
}

int main()
{
  int *p = (int *) malloc(sizeof(int));
  ordered_free02(p);
  // Here p has not been freed, but we have no way to know it
  // currently; we should assume it
  return 0;
}
