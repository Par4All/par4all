/* Check the use and translation of anywhere effect. Same as
 * pointer01.c, but the anywhere effect should be typed and not impact x,
 * y and z.
 *
 * Bug: the translation of intraprocedural points-to does not occur
 * and a reference to a points-to stub of "foo", "foo:_p_1" is
 * preserved in the effects of "main".
 */

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
  // FI: was the meaningless cast to (int *) put here with some purpose
  // double *p = (int *) malloc(sizeof(double));
  double *p = (double *) malloc(sizeof(double));
  foo(p);
  return x+y+z; // information about x, y and z should be preserved
}
