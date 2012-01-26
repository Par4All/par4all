
foo(int ** dest, int *source)
{
  *dest = source;
}


int main()
{
  int *p, *q;

  *p = 1;

  foo(&q, p);
  *q = 2;
  return 0;
}
