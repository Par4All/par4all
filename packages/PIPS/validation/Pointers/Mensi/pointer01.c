/* Example for parallelization */

#include <assert.h>

void pointer01(int n, float *p, float *q)
{
  int i;

  assert(p!=0&&q!=0);

  for(i=0; i<n; i++)
    p[i] = q[i];

  return;
}
