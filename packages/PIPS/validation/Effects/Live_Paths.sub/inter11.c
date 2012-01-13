// one call site, but the value may not be used

void foo (int *p, int value)
{
  *p = value;
}

int main()
{
  int a, b, val;
  val = 3;
  foo(&a, val);
  if ( val < 5)
      b = a;
  else
    b = val;
  return b;
}
