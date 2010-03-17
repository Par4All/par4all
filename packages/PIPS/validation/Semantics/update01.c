void foo(int j)
{
}

void update01()
{
  int i = 1;

  //i = (i+= 2);
  i += 3;
  i++;
  i--;
  foo(i);
}
