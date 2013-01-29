/* test case provided by Sven Verdoolaege, 27 January 2013
 *
 * Discussion about privatization and tilability
 *
 * here (see sven01.c) y is renamed x and x[n] is not detected as
 * privatizable because the read regions corresponding to the call to
 * g are merged and result in may regions. As a consequence in and out
 * regions are also may regions.
 */

double f(int i, int j, int k)
{
  return (double) (i+j+k);
}

double g(double x, double y)
{
  return x+y;
}

void sven02(int n, double x[n])
{
  int i, j, k;
  for (i = 0; i < n; ++i)
    for (j = 0; j < n; ++j)
      for (k = 0; k < n; ++k) {
	x[n] = f(i,j,k);
	x[i] = g(x[i], x[n]);
      }
  return;
}
