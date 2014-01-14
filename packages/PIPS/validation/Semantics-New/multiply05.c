/* Nelson Lossing observed that a large part of
 * integer_multiply_to_transformer() was never executed and that
 * correct results were obtained nevertheless.
 *
 * This may happen...
 *
 * Check squares... k >= 6*N
 */

int multiply05(int i, int j, int N)
{
  int k, l, m, n;

  i = 2;
  j = 3;
  k = i*N*j*N;
  l = i*(N*N)*j;
  m = i*j*(N*N);
  n = N*N;

  return k+l+m+n;
}
