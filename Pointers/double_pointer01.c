/* A double dereferencing should generate two levels of stubs, _a1_1 and _a1_1_1
 *
 */

void array_addition01(float**  a1)
{
  int i,j ;
  a1[i][j] = 0.;
  *a1[0] = 0;
  **a1 = 0;
  return ;
}
