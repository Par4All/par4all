/* Make sure that variables are properly renamed */

#include <stdio.h>

static int foo_0 = 3;

void flatten_code12(void)
{
  int i = foo_0;

  for(i=0;i<4;i++) {
    static int foo = 0;

    foo++;

    fprintf(stdout, "foo=%d\n", foo);
  }
}

int main()
{
  flatten_code12();

  return 0;
}
