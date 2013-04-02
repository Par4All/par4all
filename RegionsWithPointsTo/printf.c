/* Example from MYSTEP/CodeExamples/TEST_c/pointer */

#include <stdio.h>
#include <stdlib.h>


void error(const char *msg)
{
  printf("%s\n", msg);
}

void init_1D(int n, int array[n])
{
  int i;

#pragma omp for
  for (i=0; i<n; i++)
      array[i] = i;
}

void display_1D(int n, int array[n])
{
  int i;
  for (i=0; i<n; i++)
    printf("%d ", array[i]);
  printf("\n");
}

int main(void)
{
  int n=10;

  int array_1D[n];
#pragma omp parallel
  {
    init_1D(n, array_1D);

#pragma omp master
    {
      display_1D(n, array_1D);
      error("ici");
    }
  }

  return 0;
}
