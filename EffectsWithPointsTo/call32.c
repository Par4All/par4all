/* To check behavior with a free
 *
 * Same as call31, but use a structure in the caller
 *
 * Parameter i added to help debugging.
 */

#include <stdlib.h>

struct ad {int * array;};

void call32(int i, int * q)
{
  free(q);
  return;
}

int call32_caller(struct ad s)
{
  int ii = *s.array;
  call32(ii, s.array);
  return ii++; // To check indirect impact on memory effects
}
