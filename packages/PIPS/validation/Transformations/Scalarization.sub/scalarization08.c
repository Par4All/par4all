/* Expected result:
   x[i] can be scalarized, but the scalar must be copied back into x[i]

   NOTES:
   - this code is meaningless: x[i] is redefined before it is used.

 */

#include <stdio.h>
#define SIZE 10

int scalarization08(int x[SIZE], int y[SIZE][SIZE])
{
  int i,j,k;
  for(i=0 ; i < SIZE ; i++)
    for(j=0 ; j < SIZE ; j++)
      x[i] = y[i][j];
  //printf("%d", x[1]);
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

  i= scalarization08(x, y);

  printf("%d\n", i);
}
