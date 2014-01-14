/* To illustrate the interprocedural issues with free() */

#include <stdlib.h>

int ordered_free01(int *fp) {
  int *fq = fp;
  free(fp);
  fp = (int *) malloc(sizeof(int));
  return 0;
}

int main()
{
  int *p = (int *) malloc(sizeof(int));
  ordered_free01(p);
  // Here we may not know that p has been freed but we should know
  // that p may have been freed
  return 0;
}
