// external function call with pointer modified in the callee.
#include <stdlib.h>

typedef struct {float *re; float *im; } cmplx_t;

void foo(cmplx_t * p, int n)
{
  p->re = (float *) malloc(n*sizeof(float));
  p->im = (float *) malloc(n*sizeof(float));
  return;
}

int main()
{
  cmplx_t cmplxs;
  foo(&cmplxs, 10);
  return 0;
}
