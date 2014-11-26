/* Check invariant code motion with another easy case,
 * but doubly nested and invariant declaration inside loop
 */

void invariant04(int n, int ni, int nj)
{
  float x[n][n], y[n];
  int i, j;
  float s;

  for(i = 0; i<ni; i++) {
    float a;
    for(j = 0; j<nj; j++) {
      a= 10.0;
      s = a*a;
      x[i][j] = s;
    }
  }
}
