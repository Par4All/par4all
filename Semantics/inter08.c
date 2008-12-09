void foo(int i)
{
}

void inter08(int i)
{
  i = 1;
}

main()
{
  int i = 4;
  inter08(i);
  foo(i);
}
