#include <stdio.h>
#define SIZE 100

int func(int n)
{
  int x[SIZE], y[SIZE][SIZE], t[SIZE];
  int i, j;

  for (i=0 ; i < SIZE ; i++) {
    x[i] = i;
    for (j=0 ; j < SIZE ; j++)
      t[i] = x[i];
      y[i][j] = x[i] + j;
      //x[i] = t[i];
  }
  return y[n][n];
}

int main(int argc, char **argv)
{
  printf("%d\n", func(5));
}
