/* Example for dissertation by Amira Mensi */

void swap(int **p, int **q)
{
  int *pt = *p;
  *p = *q;
  *q = pt;
  return;
}

int main()
{
  int i = 1, j = 2, *pi = &i, *pj = &j;
  swap(&pi, &pj);

  return 0;
}
