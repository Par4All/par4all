// calling a function with an array argument

#include <stdio.h>
#define N 10

int foo(int p[N], int index)
{
  return p[index];
}

int main()
{
  int b[10],c;
  c = foo(b, 3);
  return c;
}
