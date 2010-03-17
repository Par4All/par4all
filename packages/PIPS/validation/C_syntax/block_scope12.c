#include <stdio.h>

int foo(int y)
{
  return y+1;
}

static int block_scope12()
{
  extern int foo(int);
  int x = 6;

  {
    extern int foo(int);
    int x = foo(x);
    printf("internal: %d\n", x);
  }

  return foo(x);
}

main()
{
  printf("external: %d\n", block_scope12());
}
