int foo(void)
{
  int foo_int = 1;
  return foo_int;
}

int bla(void)
{
  int f = foo();
  int bla_int = f + 2;
  return bla_int;
}

int main(void)
{
  int f = foo();
  int b = bla();
  printf("%d-%d\n",f,b);
  return 0;
}
