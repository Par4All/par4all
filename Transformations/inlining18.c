#include <stdio.h>

char * foo(void)
{
  const char * foo_fun = __FUNCTION__;
  return (char *) foo_fun;
}

int main(void)
{
  char * fun = foo();
  fprintf(stdout, "fun: %s\n", fun);
  fprintf(stdout, "function: %s\n", __FUNCTION__);
  return 0;
}
