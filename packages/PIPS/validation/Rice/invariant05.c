/* Check invariant code motion with another easy case,
 * but doubly nested and invariant declaration inside loop
 */

void invariant05(int n, int ni, int nj)
{
  float x[n][n], y[n];
  int i, j;
  float s;

  for(i = 0; i<ni; i++) {
    for(j = 0; j<nj; j++) {
      float a;
      a= 10.0;
      s = a*a;
      x[i][j] = s;
    }
  }
}
