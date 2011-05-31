/* Make sure that variables are properly renamed.

   This test case is derived from 'flatten_code12':

   We must ensure that moving up the declaration of variable 'foo',
   originally declared inside the for(...) loop, will not create a
   conflict with formal parameter 'foo' of function flatten_code15.
 */

#include <stdio.h>

static float foo_0 = 3.;

void flatten_code15(int foo)
{
  int i = (int) foo_0;

  foo_0 += foo;

  for(;i<7;i++) {
    static int foo = 0;
    foo++;
    fprintf(stdout, "foo=%d\n", foo);
  }
  fprintf(stdout, "foo_0=%f\n", foo_0);
}

int main()
{
  flatten_code15(1);

  return 0;
}
