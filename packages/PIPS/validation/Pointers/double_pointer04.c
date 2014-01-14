/* A double dereferencing should generate two levels of stubs, _a1_1 and _a1_1_1
 *
 */

void double_pointer04(float**  a1, int i, int j)
{
  a1[i][j] = 0.;
  return ;
}
