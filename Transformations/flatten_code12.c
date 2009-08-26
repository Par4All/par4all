/* Make sure that variables are properly renamed */

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
