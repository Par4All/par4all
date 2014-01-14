/* Check linked to schrammel05.c
 *
 * Potential bug when evaluating conditions. In fact the condition was
 * properly evaluated when false.
 *
 * However, this test case showed that squares were not analyzed as
 * sharply as they could. I improved integer_multiply_to_transformer()
 * to handle this case, but the function should be split to handle
 * squaring and sqaure detection as such.
 */

int multiply06(int i, int j, int N)
{
  int k, l, m, n;

  if(i*i>2) {
    k = i*i;
    l = k+1;
  }
  else {
    k = i*i;
    l = k+1;
  }

  return k+l+m+n;
}
