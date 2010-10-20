/* Check that property ALIASING_ACROSS_TYPES is used for
 * formal parameters and other variables.
 *
 * By default, there should be no dependence between the wto statements
 */

void formal02(int * a, float * b)
{
  int * pi;
  float * px;
  *a = 1;
  *b = 1.;
  /*
  *pi = 1;
  *px = 1.;
  */
}
