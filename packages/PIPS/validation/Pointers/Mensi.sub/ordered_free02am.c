#include<stdlib.h>

void ordered_free02(int *fp)
{
  int *fq = fp;
  fp = (int *) malloc(sizeof(int));
  free(fp);
  return;
}
