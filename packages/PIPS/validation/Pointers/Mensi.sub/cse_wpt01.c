/* Just a very basic test of Common Subexpression Elimination in the presence of pointer
 */

int cse_wpt01()
{
  int i = 0;
  int j = 1;
  int k;
  int *p = &j;
  int *q = &k;
 

  i = 2*(j+2);
  *p = *q;
  k = 3*(j+2);
}
