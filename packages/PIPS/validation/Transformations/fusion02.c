/* Test case developped for Beatrice to explore array section
 * privatization after loop fusion.
 *
 * Same as fusion01, but with wrong loop bounds so as to make fusion legal
 */

#include <stdio.h>

main()
{
  double a[100][100];
  double b[100][100];
  double c[100][100];

#define KI (3)
#define KJ (3)
  int i, j, ki, kj, n;

  scanf("%d", &n);

  for(i=0; i<n-KI;i++)
    for(j=0; j<n-KJ;j++)
      scanf("%f", &a[i][j]);
  for(i=0; i<n-KI;i++)
    for(j=0; j<n-KJ;j++) {
      b[i][j] = 0.;
      for(ki=0;ki<KI;ki++)
	b[i][j] += a[i+ki][j];
    }
  for(i=0; i<n-KI;i++)
    for(j=0; j<n-KJ;j++) {
      c[i][j] = 0.;
      for(kj=0;kj<KJ;kj++)
	c[i][j] += b[i][j+kj];
    }
  for(i=0; i<n-KI;i++)
    for(j=0; j<n-KJ;j++)
      //printf("%f", &c[i][j]); -> issues with conflict computation
      c[i][j]++;
}
