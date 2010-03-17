/* complex expressions with side effects */

void foo(int i)
{
  ;
}

void assign02()
{
  int i = 1;
  int j;
  int k;
  int n;

  n++;
  j = k = 2;
  /* i is assigned before k, left-to-right evaluation */
  i = j, k = i;

  j = 2 * i, i = j + 1;

  foo(i);
}
