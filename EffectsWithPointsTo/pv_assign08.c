// rhs with an external call -> correct but unprecise result for the moment

#include <stdlib.h>

int * foo()
{
  return( (int *) malloc (sizeof(int) ));
}

int main()
{
  int *a;
  a = foo();
  return(0);
}
