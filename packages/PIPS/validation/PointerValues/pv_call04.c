// external function call with pointer modified in the callee.
#include <stdlib.h>

void foo(int **q, int *p)
{
  *q = p;
}

int main()
{
  int a = 1, *p, **q;
  p = &a;
  q = (int **) malloc(sizeof(int *));
  foo(q, p);
  return 0;
}
