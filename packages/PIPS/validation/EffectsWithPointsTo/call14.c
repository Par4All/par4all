/* Linearization of call04: 2-D array y is initialized as a 1-D array
 * using pointer x.
 *
 * Bug: the points-to stub _x_3 is not translated back into y[*][*]
 */

#include<stdio.h>

void call14(int n, int m, int *x)
{
  int i;

  for(i=0; i<n*m;i++)
      x[i] = i;
  return;
}

int main()
{
  int d1 = 4;
  int d2 = 4;
  int y[d1][d2];
  int i, j;

  call14(d1, d2, (int *) y);

  for(i=0;i<d1;i++) {
    for(j=0;j<d2;j++)
      printf("y[%d][%d] = %d\t", i, j, y[i][j]);
    printf("\n");
  }
  return 0;
}
