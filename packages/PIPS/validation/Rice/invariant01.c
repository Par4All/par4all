/* Check invariant code motion with an easy case
 */

void invariant01(int n, int ni, int nj)
{
  float x[n][n], y[n];
  int i, j;
  float s;

  for(i = 0; i<ni; i++)
    for(j = 0; j<nj; j++) {
      s = y[i];
      x[i][j] = s;
    }
}
