/* Make sure that static variables are not reinitialized */

#include <stdio.h>

void flatten_code11(void)
{
  int i;

  /* Let's execute this loop four times! */
  for(i=0;i<4;i++) {
    static int foo = 0;

    /* Let's update a static variable! */
    foo++;

    fprintf(stdout, "foo=%d\n", foo);
  }
}

int main()
{
  flatten_code11();

  return 0;
}
