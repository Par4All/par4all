int foo(int *a)
{
  return *a;
}

int main()
{
  int i;
  foo(&i);
  return i;
}
