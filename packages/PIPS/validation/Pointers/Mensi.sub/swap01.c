void swap(int **p, int **q)
{
  int *pt = *p;
  *p = *q;
  *q = pt;
  return;
}

int main()
{
  int i = 1, j = 2, *pi = &i, *pj = &j, **ppi = &pi, **ppj = &pj;
  swap(ppi, ppj);

  return 0;
}
