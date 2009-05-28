#include <stdio.h>

int func(int n)
{
  int x[n], y[n][n], t[n];
  int i, j;

  for (i=0 ; i < n ; i++) {
    x[i] = i;
    for (j=0 ; j < n ; j++) {
      y[i][j] = x[i] ^ 2;
      y[i][j] = x[i] + j;
    }
  }
  return y[n][n];
}

int main(int argc, char **argv)
{
  printf("%d\n", func(5));
}
