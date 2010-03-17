/* Make sure that external variables are not renamed, here "stderr"
   and "foo" */

#include <stdio.h>

void flatten_code10(void)
{
  int i = 2;

  {
    extern int foo;

    fprintf(stderr, "Hello world!\n");
  }
}
