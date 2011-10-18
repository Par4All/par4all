// scalarizable reference passed as argument to a call -> scalarize

#include <stdio.h>

#define N 10

void foo(int tmp, int out[N], int index)
{
  out[index] = tmp;
}

int main()
{
  int a[10], b[10], i;

  for(i = 0; i< 10; i++)
    {
      a[i] = i;
      foo(a[i], b, i);
    }

  for(i = 0; i< 10; i++)
    {
      printf("%d\n", b[i]);
    }

  return 0;
}
