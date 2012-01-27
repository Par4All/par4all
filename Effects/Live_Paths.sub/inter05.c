// calling a function with a sub-array argument

#include <stdio.h>
#define N 10

int foo(int p[5])
{
  return p[3];
}

int main()
{
  int b[10],c;
  c = foo(&b[5]);
  return c;
}
