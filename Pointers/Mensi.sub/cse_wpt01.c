/* Just a very basic test of Common Subexpression Elimination in the presence of pointer
 */

int cse_wpt01()
{
  int i = 0, j = 1, k;
  int *p = &j;
  int *q = &k;
  i = 2*(j+2);
  *q = *p;
  k = 3*(j+2);
}
