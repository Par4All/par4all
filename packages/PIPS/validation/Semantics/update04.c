/* make sure that the current precondition is properly used when computing transformers */

void foo(int j)
{
}

void update04()
{
  int i = 1;
  int j, k;

  //i = 2;
  j = ++i;
  k = i++;
  foo(i);
}
