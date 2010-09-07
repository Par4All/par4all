// function that differ from a case.

int foo(void)
{
  return 1;
}

int Foo(void)
{
  return 10;
}

int FOO(void)
{
  return 100;
}

int Main(void)
{
  return foo()+Foo()+FOO()+1000;
}

int main(void)
{
  return Main()+10000;
}
