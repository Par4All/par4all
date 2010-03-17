/* Check label handling in loop interchange...
 *
 * The initial label must be preserved, i.e. switch from one loop to another
 */

void loop_interchange01(int n, int ni, int nj)
{
  float x[n][n];
  int i, j;

 l1:  for(i = 0; i<ni; i++)
  l2:   for(j = 0; j<nj; j++)
      x[i][j] = 0.;
}
