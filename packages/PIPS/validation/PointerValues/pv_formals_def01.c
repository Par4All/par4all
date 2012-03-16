int foo(int *a)
{
  *a = 1;
  return *a;
}

int main()
{
  int i;
  foo(&i);
  return i;
}
