/* Nelson Lossing observed that a large part of
 * integer_multiply_to_transformer() was never executed and that
 * correct results were obtained nevertheless.
 *
 * This may happen...
 *
 * Bug found in a new function called by Nelson in
 * integer_multiply_to_transformer() as an improvement...
 */

int multiply02(int i, int j, int N)
{
  int k;

  i = 2;
  j = 3;
  k = i*N*j;
  return k;
}
