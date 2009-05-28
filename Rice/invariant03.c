/* Check invariant code motion with another easy case, but doubly nested
 */

void invariant03(int n, int ni, int nj)
{
  float x[n][n], y[n];
  int i, j;
  float s, a;

  for(i = 0; i<ni; i++)
    for(j = 0; j<nj; j++) {
      s = a*a;
      x[i][j] = s;
    }
}
