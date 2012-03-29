// external function call with no pointer involved

#include <stdio.h>

void foo(int a)
{
  printf("%d\n", a);
}

int main()
{
  int a = 1;
  foo(2);
  foo(a);
  return 0;
}
