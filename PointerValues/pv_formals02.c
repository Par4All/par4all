int foo(int **a)
{
  return **a;
}

int main()
{
  int res;
  int p;
  int *q;
  q = &p;
  res = foo(&q);
  return res;
}
