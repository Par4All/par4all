// several call sites

void foo (int *p, int value)
{
  *p = value;
}

int main()
{
  int a, b, val;
  val = 3;
  foo(&a, val);
  foo(&b, val+1);
  return a+b;
}
