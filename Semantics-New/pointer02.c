// Check the use and translation of anywhere effect. Same as
// pointer01.c, but the anywhere effect should be typed and not impact x,
// y and z.

#include <assert.h>

int x;
int y;
int z;

void foo(double *p)
{
  assert(p!=0);
  (*p)++;
}

int main()
{
  x= 1, y=2, z=3;
  double *p = (int *)malloc(sizeof(double));
  foo(p);
  return x+y+z; // information about x, y and z should be preserved
}
