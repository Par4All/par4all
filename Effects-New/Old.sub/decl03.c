#include <stdio.h>
int i;

void foo()
{
  int j =i;
  printf("%d",j);
}

int main()
{
  i = 3;
  foo();
  return 0;
}
