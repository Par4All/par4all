/* Check label handling in loop tiling for three loops...
 *
 * The initial label must be preserved, but not the other ones
 */

void loop_tiling03(int n, int ni, int nj, int nk)
{
  float x[n][n][n];
  int i, j, k;

 l1:  for(i = 0; i<ni; i++)
  l2:   for(j = 0; j<nj; j++)
    l3:   for(k = 0; k<nk; k++)
	x[i][j][k] = 0.;
}
