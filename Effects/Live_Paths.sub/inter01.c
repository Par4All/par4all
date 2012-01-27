#include <stdio.h>

void foo(int *p)
{
  *p = 10;
}

int main()
{
  int a;
  a=10;

  foo(&a);
  return a;
}
