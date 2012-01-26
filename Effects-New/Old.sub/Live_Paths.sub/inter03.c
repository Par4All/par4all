// calling a function with a return value used array indices

#include <stdio.h>

int foo(int *p)
{
  return (*p)+1;
}

int main()
{
  int a,b[10],c;
  c = b[foo(&a)];
  return c;
}
