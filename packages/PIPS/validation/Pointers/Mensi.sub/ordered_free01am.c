#include<stdlib.h>

void ordered_free01(int *fp)
{
  int *fq = fp;
  free(fp);
  fp = (int *) malloc(sizeof(int));
  return;
}
