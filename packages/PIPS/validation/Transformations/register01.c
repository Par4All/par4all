int dixsept(int i)
{
  int l = i+17, m = 2*i+13;
  return l+m;
}

int register01(int i, int * p)
{
  int j = i, k = 1;
  *p = j;
  p = &k;
  return dixsept(k);
}
