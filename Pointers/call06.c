/* #include<stdio.h> */
#include<stdlib.h>

void call06(int r, int m, int *x)
{
  int j;

  for(j=0; j<m;j++)
    x[j] = 10*r+j;
  return;
}

main()
{
  int d1 = 4;
  int d2 = 4;
  int y[d1][d2];
  int i, j;

  for(i=0;i<d1;i++)
    call06(i, d2, &(y[i][0]));
  exit(0);
}
