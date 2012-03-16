#include <stdlib.h>

typedef struct {int a;} * my_struct_p;

int foo(my_struct_p p)
{
  return p->a;
}

int main()
{
  int res;
  my_struct_p q;
  q = (my_struct_p) malloc(sizeof( my_struct_p *));
  res = foo(q);
  return res;
}
