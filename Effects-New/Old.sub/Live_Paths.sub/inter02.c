// calling a function with a return value used in assignment

#include <stdio.h>

int foo(int *p)
{
  *p = 10;
  return *p+1;
}

int main()
{
  int a,b;
  a=10;

  b= foo(&a);
  return b;
}
