#include <stdlib.h>
#include <assert.h>

void pointer10(double (*a)[10], double (*b)[10],double (*c)[10], int N)
{
  double (*w)[N];
  int i;

  assert(a!=NULL && b!=NULL && c!=NULL); 

  w = (double (*)[N]) malloc(N * sizeof(double));

  for (i = 0; i < N; i++)
    (*w)[i]  = (*c)[i] + (*a)[i] * (*b)[i];

  return;
}
