void foo(int j)
{
}

void update02()
{
  int i = 1;
  int j;

  //i = (i+= 2);
  j = (i+= 2);
  j = i++;
  j = i--;
  j = ++i;
  j = --i;
  foo(j);
}
