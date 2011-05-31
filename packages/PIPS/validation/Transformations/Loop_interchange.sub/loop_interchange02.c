/* Check label handling in loop interchange...
 *
 * The initial label must be preserved, i.e. switch from one loop to another
 */

void loop_interchange02(int n, int ni, int nj, int nk)
{
  float x[n][n][n];
  int i, j, k;

 l1:  for(i = 0; i<ni; i++)
  l2:   for(j = 0; j<nj; j++)
  l3:   for(k = 0; k<nk; k++)
      x[i][j][k] = 0.;
}
