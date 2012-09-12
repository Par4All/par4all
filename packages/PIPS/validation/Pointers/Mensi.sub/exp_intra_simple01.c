int foo(int **p, int *q)
{
  int *i, j=0;
  
  i = *p;
  p = &i;
  j = *q;
  i = &j;
  q = *p;
  
  return j;
}
