void foo(int j)
{
}

void update03()
{
  int i = 1;

  /* This does not make much sense, just like "i = i;". */
  i = (i+= 2);
  i = ((i+= 2)+1);
  foo(i);
}
