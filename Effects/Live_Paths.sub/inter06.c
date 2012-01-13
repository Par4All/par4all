// calling a function with a sub-array argument

#include <stdio.h>
#define N 10

int foo(int p[5], int index)
{
  return p[index];
}

int main()
{
  int b[10],c;
  c = foo(&b[5], 3);
  return c;
}
