/* Check that property ALIASING_ACROSS_FORMAL_PARAMETER is used for
 * formal parameters.
 *
 * By default, there should be no dependence between the wto statements
 */

void formal01(int * a, int * b)
{
  *a = 1;
  *b = 1;
}
