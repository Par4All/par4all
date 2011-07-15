#include <stdlib.h>
#include <stdio.h>

#define N 10
//comparison function

typedef struct{ int *n;} s_int;

int foo(const void*pa1, const void* pa2)
{
  int a1 = *(((s_int *)pa1)->n);
  int a2 = *(((s_int *)pa2)->n);
  if (a1 < a2) return -1;
  else if (a1 > a2) return +1;
  return 0;
}

int main()
{
  int b[N];
  s_int a[N];
  int i;
  for (i= 0; i<N; i++)
    {
      b[i] = N-i-1;
      a[i].n = &b[i];
    }
  for(i=0; i<N; i++)
    printf("%d\n", *a[i].n);
  qsort(a, N, sizeof(s_int), &foo);
  for(i=0; i<N; i++)
    printf("%d\n", *a[i].n);
  return 0;
}
