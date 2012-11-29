/* To check behavior with a free
 *
 * Same as call33, but use an array in the callee
 *
 * Parameter i added to help debugging.
 */

#include <stdlib.h>

void call35(int i, int * q[10])
{
  *q[0] = 1; // added to check effects
  free(q[i]);
  return;
}

int call35_caller(int * qq[10])
{
  int ii = *qq[0];
  call35(ii, qq);
  return ii++; // To check indirect impact on memory effects
}
