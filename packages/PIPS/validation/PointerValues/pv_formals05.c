#include <stdlib.h>

typedef struct {int a;} my_struct;

int foo(my_struct * p)
{
  return p->a;
}

int main()
{
  int res;
  my_struct * q;
  q = (my_struct *) malloc(sizeof( my_struct));
  res = foo(q);
  return res;
}
