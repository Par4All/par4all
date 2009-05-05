/* complex expressions with side effects */

void foo(int i)
{
  ;
}

void assign03()
{
  int i;
  int j;
  int k;
  int n;

  j = 2; 
  i = j + (j = 0) + j;

  foo(i);
}
