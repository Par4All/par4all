// scalarizable reference through a call -> do not scalarize

#include <stdio.h>

#define N 10

void foo(int tmp[N], int out[N], int index)
{
  tmp[index] = index;
  out[index] = tmp[index];
}

int main()
{
  int a[10], b[10], i;

  for(i = 0; i< 10; i++)
    {
      foo(a, b, i);
    }

  for(i = 0; i< 10; i++)
    {
      printf("%d\n", b[i]);
    }

  return 0;
}
