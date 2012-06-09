void f(int **p, int **q, int **r)
{
  *p = *q;
  *q = *r;
  return;
}

int main()
{
  int x, y, z;
  int *x0 = &x, *y0 = &y, *z0 = &z;
  
  f(&x0, &y0, &x0);

  return 0;
}
