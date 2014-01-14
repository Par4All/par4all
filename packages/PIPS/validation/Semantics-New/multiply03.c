/* Nelson Lossing observed that a large part of
 * integer_multiply_to_transformer() was never executed and that
 * correct results were obtained nevertheless.
 *
 * This may happen...
 *
 * Same as multiply02, but transformers are computed in context
 */

int multiply03(int i, int j, int N)
{
  int k;

  i = 2;
  j = 3;
  k = i*N*j;
  return k;
}
