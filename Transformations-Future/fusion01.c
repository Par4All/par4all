/* Test case developped for Beatrice to explore array section
 * privatization after loop fusion.
 *
 * We could reuse the vbl (sonar application), but let's start with a
 * smaller example...
 *
 * Because of the computation, the loop bounds are not equal. However,
 * the last two loop nests should be fused.
 *
 * Loop fusion could be extended to handled included iteration sets
 * instead of equal iteration sets.
 */

#include <stdio.h>

main()
{
  double a[100][100];
  double b[100][100];
  double c[100][100];

  int i, j, ki, kj, n;

  scanf("%d", &n);

  for(i=0; i<n;i++)
    for(j=0; j<n;j++)
      scanf("%f", &a[i][j]);
  for(i=0; i<n-ki;i++)
    for(j=0; j<n;j++) {
      b[i][j] = 0.;
      for(ki=0;ki<3;ki++)
	b[i][j] += a[i+ki][j];
    }
  for(i=0; i<n-ki;i++)
    for(j=0; j<n-kj;j++) {
      c[i][j] = 0.;
      for(kj=0;kj<3;kj++)
	c[i][j] += b[i][j+kj];
    }
  for(i=0; i<n-ki;i++)
    for(j=0; j<n-kj;j++)
      //printf("%f", &c[i][j]); -> issues with conflict computation
      c[i][j]++;
}
