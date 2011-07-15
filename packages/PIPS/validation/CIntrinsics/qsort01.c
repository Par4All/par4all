#include <stdlib.h>
#include <stdio.h>

#define N 10
//comparison function

int foo(const void*pa1, const void* pa2)
{
  int a1 = *(int *)pa1;
  int a2 = *(int *)pa2;
  if (a1 < a2) return -1;
  else if (a1 > a2) return +1;
  return 0;
}

int main()
{
  int a[N];
  int i;
  for (i= 0; i<N; i++)
    a[i] = N-i-1;
  qsort(a, N, sizeof(int), &foo);
  for(i=0; i<N; i++)
    printf("%d\n", a[i]);
  return 0;
}
