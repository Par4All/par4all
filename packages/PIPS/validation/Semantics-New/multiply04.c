/* Nelson Lossing observed that a large part of
 * integer_multiply_to_transformer() was never executed and that
 * correct results were obtained nevertheless.
 *
 * This may happen...
 *
 * Same as multiply03, but with intervals
 */

#include <stdlib.h>
#include <assert.h>

int multiply04(int i, int j, int N)
{
  int k, l;

  assert(N>=1);
  assert(2<=i && i<=3);
  assert(5<=j && j<=10);

  k = i*N*j;
  l = i*N*j*N;
  return k+l;
}
