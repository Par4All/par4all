#include <stdio.h>

void display_1D(int n, int array[n])
{
  int i, j;
   for (i=0; i<n; i++)
     printf("%d", array[i]);
}

void init_2D(int n, int m, int array[n][m])
{
  int i, j;

  for (i=0; i<n; i++)
    for (j=0; j<m; j++)
      array[i][j] = i+j;
}

int main(void)
{
  int n, m;
  scanf("%d%d", &n, &m);

  int size= n*m;
  int array_1D[size];

  init_2D(n, m, array_1D);

  display_1D(size, array_1D);

  return 0;
}
