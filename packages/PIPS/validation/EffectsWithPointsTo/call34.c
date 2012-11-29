/* To check behavior with a free
 *
 * Same as call32, but use the structure in the callee
 *
 * Parameter i added to help debugging.
 */

#include <stdlib.h>

struct ad {int * array;};

void call34(int i, struct ad s)
{
  free(s.array);
  return;
}

int call34_caller(struct ad s)
{
  int ii = *s.array;
  call34(ii, s);
  return ii++; // To check indirect impact on memory effects
}
