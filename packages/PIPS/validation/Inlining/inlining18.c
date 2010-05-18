char * foo(void)
{
  const char * foo_fun = __FUNCTION__;
  return (char *) foo_fun;
}

int main(void)
{
  const char * fun, * fun2, * fun3;
  fun = foo();
  fun2 = __FUNCTION__;
  fun3 = __func__;
  return 0;
}
