/* Check new variable declarations and label handling (preserve l1, destroy l2)
 */

void loop_hyperplane01(int n, int ni, int nj)
{
  float x[n][n];
  int i, j;

 l1:  for(i = 0; i<ni; i++)
  l2:   for(j = 0; j<nj; j++)
      x[i][j] = 0.;
}
