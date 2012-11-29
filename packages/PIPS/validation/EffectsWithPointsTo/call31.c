/* To check behavior with a free
 *
 * Simplified version of call30: q is expliclty dereferenced in call31_caller
 * to make sure that the on-demand mechanism is not too stressed by
 * the call.
 *
 * Parameter i added to help debugging.
 */

#include <stdlib.h>

void call31(int i, int * q)
{
  free(q);
  return;
}

int call31_caller(int * qq)
{
  int ii = *qq;
  call31(ii, qq);
  return ii++; // To check indirect impact on memory effects
}
