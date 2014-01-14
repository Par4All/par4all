/* test case provided by Sven Verdoolaege, 27 January 2013
 *
 * Discussion about privatization and tilability
 *
 * If y is renamed x, is x[n] detected as privatizable? -> No (see sven02 test case)
 *
 */

double f(int i, int j, int k)
{
  return (double) (i+j+k);
}

double g(double x, double y)
{
  return x+y;
}

void sven01(int n, double x[n])
{
  int i, j, k;
  double y[n];
  for (i = 0; i < n; ++i)
    for (j = 0; j < n; ++j)
      for (k = 0; k < n; ++k) {
	y[n] = f(i,j,k);
	x[i] = g(x[i], y[n]);
      }
  return;
}
