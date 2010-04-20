/* Check that typedef do not perturb simplify_C_expression and later
   partial_eval */

typedef int foo;

void partial_eval03()
{
  foo size = 2;
  foo i;

  i = size;
}
