/* complex expressions with redundant side effects */

void foo(int j)
{
  j++;
}

void assign04()
{
  int i;

  // This assignment works fine with transformers
  i =  (i = 2) + 1;

  // Go around the problem with user_call_to_transformer()
  foo( i =  (i = 2) + 1);

  // As in:
  i =  (i = 2) + 1;
  foo(i);
}
