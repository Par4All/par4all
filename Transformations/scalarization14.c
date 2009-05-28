/* Expected result:
   x[i] and t[i] can be scalarized

   NOTES:
   - no copy-out on x[i] and t[i], as they are not copied out. This is OK.
   - Additionally, one too many scalar is declared (__ld__3)
 */

#include <stdio.h>
#define SIZE 100

int func(int n)
{
  int x[SIZE], y[SIZE][SIZE], t[SIZE];
  int i, j;

  for (i=0 ; i < SIZE ; i++) {
    x[i] = i;
    for (j=0 ; j < SIZE ; j++) {
      t[i] = x[i];
      y[i][j] = x[i] + j + t[i];
    }
  }
  i = y[n][n] + y[0][0] + y[0][n] + y[n][0];
  return i;
}

int main(int argc, char **argv)
{
  printf("%d\n", func(5));
}
