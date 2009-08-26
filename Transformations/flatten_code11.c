/* Make sure that static variables are not reinitialized */

#include <stdio.h>

void flatten_code11(void)
{
  int i;

  for(i=0;i<4;i++) {
    static int foo = 0;

    foo++;

    fprintf(stdout, "foo=%d\n", foo);
  }
}

int main()
{
  flatten_code11();

  return 0;
}
