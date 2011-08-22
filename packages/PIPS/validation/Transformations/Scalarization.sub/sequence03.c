/* Check privatization in sequences and its control by thresholding
 *
 * Issue: if x is not copy-in nor copy-out, privatization always occur
 *
 * Thresholding to allow scalarization of x[1] but not of y[1][1]
 */

#include <stdio.h>
#define SIZE 10

int sequence03(int x[SIZE], int y[SIZE][SIZE])
{
  int k;
  x[1] = x[1] + y[1][1];
  x[1] = x[1] + y[1][1];
  x[1] = x[1] + y[1][1];
  x[1] = x[1] + y[1][1];
  k = x[1];
  return k;
}

int main(int argc, char **argv)
{
  int i, j;
  int x[SIZE], y[SIZE][SIZE];

  for (i=0 ; i < SIZE ; i++)
    for (j=0 ; j < SIZE ; j++)
      //scanf("%d", &y[i][j]);
      y[i][j] = 100*i+j;

  i= sequence03(x, y);

  printf("%d\n", i);
}
