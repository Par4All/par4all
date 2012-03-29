// external function call with pointer dereferenced in the callee.

#include <stdio.h>

void foo(int *q)
{
  printf("%d\n", *q);
}

int main()
{
  int a = 1, *p;
  p = &a;
  foo(p);
  return 0;
}
