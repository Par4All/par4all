int* foo()
{
  int cond = 0, i = 1, j = 2;
  int *p, *q;

  p = &i;
  q = &j;

  if(cond)
    return p;
  else
    return q;

  q = &cond;
  return q;
}

int main()
{
  int *r;
  r = foo();

  /* r's value is indeterminate and the dereferencement is forbidden
   * by the standard.
   */
  return *r;
}
