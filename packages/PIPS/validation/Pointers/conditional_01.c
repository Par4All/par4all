int conditional_01()
{
  int *q;
  int *p;
  int i=0, j=1;
  q = &i;
  p = i>0 ? &j : q;
  return 0;
}
