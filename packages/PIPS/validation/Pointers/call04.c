/* #include<stdio.h> */

void call04(int n, int m, int x[n][m])
{
  int i, j;

  for(i=0; i<n;i++)
    for(j=0; j<m;j++)
      x[i][j] = 0;
}

int main()
{
  int d1 = 10;
  int d2 = 10;
  int y[d1][d2];

  call04(d1, d2, y);
  return 0;
}
