/* test case provided by Sven Verdoolaege, 27 January 2013
 *
 * Discussion about privatization and tilability
 *
 * Forward substitution of x[n] is not applied. Is it only applied to
 * scalar variables?
 * 
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

void sven04(int n, double x[n+1])
{
  int i, j, k;
  //double t;
  for (i = 0; i < n; ++i)
    for (j = 0; j < n; ++j)
      for (k = 0; k < n; ++k) {
	x[n] = f(i,j,k);
	x[i] = g(x[i], x[n]);
      }
  for(i=0; i<n; ++i)
    printf("%f ", x[i]);
  return;
}
