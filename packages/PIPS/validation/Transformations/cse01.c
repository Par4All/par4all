/* Just a very basic test of Common Subexpression Elimination
 */

int cse01()
{
  int i;
  int j;
  int k;

  i = 2*(j+2);
  k = 3*(j+2);
}
