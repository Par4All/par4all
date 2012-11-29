#include<stdio.h> 

void call05(int r, int m, int *x)
{
  int j;

    for(j=0; j<m;j++)
      x[j] = 10*r+j;
    return;
}

int main()
{
  int d1 = 4;
  int d2 = 4;
  int y[d1][d2];
  int i, j;

  for(i=0;i<d1;i++)
    call05(i, d2, y[i]);

  for(i=0;i<d1;i++) {
    for(j=0;j<d2;j++)
      printf("y[%d][%d] = %d\t", i, j, y[i][j]);
    printf("\n");
  }
  return 0;
}
