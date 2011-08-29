int foo1(void)
{
  extern int bla(int);
  return bla(3);
}

int foo2(void)
{
  extern int bla(int);
  return bla(7);
}

int inlining31(void)
{
  int i1, i2;
  i1 = foo1();
  i2 = foo2();
  return i1+i2;
}
