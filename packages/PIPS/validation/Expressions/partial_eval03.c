/* Check that typedef do not perturb simplify_C_expression and later
   partial_eval */

#include <stdio.h>

typedef int foo;

foo partial_eval03()
{
  foo size = 2;
  foo i;

  i = size;
  return i;
}
main()
{
    printf("%d",partial_eval03());
}

