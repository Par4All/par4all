// external function call with pointers involved in the caller only

#include <stdio.h>

void foo(int a)
{
  printf("%d\n", a);
}

int main()
{
  int a = 1, *p;
  p = &a;
  foo(*p);
  foo(a);
  return 0;
}
