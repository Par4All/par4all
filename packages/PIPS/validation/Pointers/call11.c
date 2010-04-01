/* #include<stdio.h> */

void call11(int n, int m, int x[n][m])
{
  int i, j;

  for(i=0; i<n;i++)
    for(j=0; j<m;j++)
      x[i][j] = 0;
}

main()
{
  int d1 = 10;
  int d2 = 10;
  int d3 = 10;
  int d4 = 10;
  int y[d1][d2][d3][d4];
  int i, j;

  for(i=0; i<d1;i++)
    for(j=0; j<d2;j++)
      call11(d3, d4, &y[i][j][0]);
}
