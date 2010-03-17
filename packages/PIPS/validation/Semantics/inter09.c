void foo(int i)
{
}

int inter09(int i)
{
  return i+1;
}

main()
{
  int i = 4;
  i = inter09(i);
  foo(i);
}
