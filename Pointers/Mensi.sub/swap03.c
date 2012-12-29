/* Equivalent to swap01, but analyzed with the fast interprocedural
   analysis */

void swap03(int **p, int **q)
{
  int *pt = *p;
  *p = *q;
  *q = pt;
  return;
}

int main()
{
  int i = 1, j = 2, z = 3, *pi = &i, *pj = &j, *pz = &z,
    **ppi = &pi, **ppj = &pj;
  swap03(ppi, ppj);

  return 0;
}
