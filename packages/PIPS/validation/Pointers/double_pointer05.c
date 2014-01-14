/* A double dereferencing should generate two levels of stubs, _a1_1 and _a1_1_1
 *
 * Occurence in a condition
 */

int double_pointer05(float**  a1, int i, int j)
{
  int c = 1;
  if(a1[i][j] == 0.)
    c = 0;
  return c;
}
