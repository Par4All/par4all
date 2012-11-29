/* To check behavior with a free
 *
 * Same as call31, but use an array in the caller
 *
 * Parameter i added to help debugging.
 */

#include <stdlib.h>

void call33(int i, int * q)
{
  *q = 1; // added to check effects
  free(q);
  return;
}

int call33_caller(int * qq[10])
{
  int ii = *qq[0];
  call33(ii, qq[ii]);
  return ii++; // To check indirect impact on memory effects
}
