#include <stdio.h>

int main(void)
{
  int array[10];
  int i;

#pragma omp parallel for private(i)
  for(i=0; i<10; i++)
    {
      array[i]=i;
    }

  for(i=0; i<10; i++)
    {
      printf("%d ",array[i]);
    }

  printf("\n");
  return 0;
}
