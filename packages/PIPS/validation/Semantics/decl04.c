void foo(int x)
{
}

void decl04()
{
  int i = 1;

  {
    int j = i;

    i = i + 1;

    foo(j);
  }

  foo(i);
}
