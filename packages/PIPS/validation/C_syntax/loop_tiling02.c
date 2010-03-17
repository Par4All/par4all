/* Check PIPS behavior when a variable is not declared.
 *
 * The parser makes it a function by default although its occurences
 * do not fit a function call or a function pointer assignment...
 */

void loop_tiling02(int ni, int nj, int nk)
{
  float x[n][n][n];
  int i, j, k;

 l1:  for(i = 0; i<ni; i++)
  l2:   for(j = 0; j<nj; j++)
  l3:   for(k = 0; k<nk; k++)
      x[i][j][k] = 0.;
}
