/* The printf() is useful to check the scoping used by gcc, but the
   resulting compilation unit makes debugging a pain.*/

#include <stdio.h>

static void foo(void)
{
  printf("call to local function\n");
}

main()
{
  foo();
  {
    extern void foo();

    foo();
  }
}
