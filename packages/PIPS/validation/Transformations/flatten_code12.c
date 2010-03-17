/* Make sure that variables are properly renamed.

   We must ensure that moving up the declaration of variable 'foo',
   will not create a conflict with variable 'foo_0' declared just
   above function flatten_code12: 'foo' should be renamed into
   'foo_1', not 'foo_0'.
 */

#include <stdio.h>

static float foo_0 = 3.;

void flatten_code12(void)
{
  int i = (int) foo_0;

  foo_0++;

  for(;i<7;i++) {
    static int foo = 0;

    foo++;

    fprintf(stdout, "foo=%d\n", foo);
  }
  fprintf(stdout, "foo_0=%f\n", foo_0);
}

int main()
{
  flatten_code12();

  return 0;
}
