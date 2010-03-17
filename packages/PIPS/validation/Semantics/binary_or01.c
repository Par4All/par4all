/* check behavior of binary or*/

# include <stdio.h>

int foo(void)
{
  fprintf(stderr, "foo is called and executed\n");

  return 0;
}

binary_or01()
{
  int ret = 1;

  ret |= foo();

  ret = ret || foo();
}
