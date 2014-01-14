#include <stdio.h>

#define N 100000

int main(int argc, char **argv)
{
  int i;
  int a[N], b[N], c[N], d[N];

  for (i=0; i < N; i++)
    {
      b[i] = 2;
      c[i] = 5;
    }

#pragma omp parallel
  {
#pragma omp for
    for (i = 0; i < N; i++)
      a[i] = b[i] + c[i];

#pragma omp for
  for (i = 0; i < N; i++)
    d[i] = a[i] + b[i];
  }

  printf("a[%d] = %d\n", 1, a[1]);
  printf("d[%d] = %d\n", 5, d[5]);

  return 0;
}
