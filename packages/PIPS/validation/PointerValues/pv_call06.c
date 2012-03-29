// external function call returning a pointer.


int *foo(int *q)
{
  return q;
}

int main()
{
  int a = 1, *p;
  p = foo(&a);
  return 0;
}
