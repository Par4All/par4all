int foo1(void)
{
  extern int bla(void);
  return bla();
}

int foo2(void)
{
  extern int bla(void);
  return bla();
}

int flatten_code13(void)
{
  int i1, i2;
  i1 = foo1();
  i2 = foo2();
  return i1+i2;
}
