#include <stdio.h>
#include <stdlib.h>

#define N 100

int main(int argc, char *argv[])
{
  int i;
  int sum = 0;
  double A[N];
 
  for (i = 0; i < N; i++) {
    A[i] = 1;
  }
 
#pragma omp parallel for reduction(+:sum) 
  for (i = 0; i < N; i++) {
    sum = sum + A[i];
  }

  printf("sum = %d\n", sum);
  return EXIT_SUCCESS;
}
