/* Check invariant code motion with an easy case
 */

void invariant02(int ni, int nj)
{
  int n = 10;
  float y[n];
  int i, j;
  float s, a, b;

  for(i = 0; i<ni; i++) {
    s = a*b;
    y[i] = s;
  }
}
