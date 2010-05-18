void foo(int i)
{
  i = i + 1;
  return;
}

void inlining20(int j)
{
  foo(2);
  foo(j);
}
