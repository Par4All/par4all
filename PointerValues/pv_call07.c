// external function call returning a pointer.
#include <stdlib.h>

int *foo(int n)
{
  int * res;
  res = (int *) malloc(n * sizeof(int));
  return res;
}

int main()
{
  int *p;
  p = foo(10);
  return 0;
}
