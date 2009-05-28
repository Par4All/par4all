/* Expected result:
   x[i] and t[i] can be scalarized

   NOTES:
   - no copy-out on x[i] and t[i], as they are not copied out. This is OK.
   - Additionally, one too many scalar is declared (__ld__3)
 */

#include <stdio.h>

int func(int n)
{
  int x[n], y[n][n];
  int i, j;

  for (i=0 ; i < n ; i++) {
    x[i] = i;
    for (j=0 ; j < n ; j++) {
      y[i][j] = x[i] ^ 2;
      y[i][j] = x[i] + j;
    }
  }
  return y[n-1][n-1];
}

int main(int argc, char **argv)
{
  printf("%d\n", func(5));
}
