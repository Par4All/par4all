/* test case provided by Sven Verdoolaege, 27 January 2013
 *
 * Discussion about privatization and tilability
 *
 * If y is renamed x, is x[n] privatizable? No, but it is scalarizable.
 *
 * PIPS however cannot find about it because the regions for one array
 * are merged.
 *
 * Bug: y is privatized but at the wrong level, outside of the loop
 */

#include <stdio.h>

double f(int i, int j, int k)
{
  return (double) (i+j+k);
}

double g(double x, double y)
{
  return x+y;
}

void sven03(int n, double x[n+1])
{
  int i, j, k;
  //double t;
  for (i = 0; i < n; ++i)
    for (j = 0; j < n; ++j)
      for (k = 0; k < n; ++k) {
	double t = f(i,j,k);
	//t = f(i,j,k);
	x[n] = t;
	x[i] = g(x[i], t);
      }
  for(i=0; i<n; ++i)
    printf("%d ", x[i]);
  return;
}
