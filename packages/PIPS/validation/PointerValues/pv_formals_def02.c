int foo(int **a)
{
  *a = (int *) malloc(sizeof(int));
  **a = 1;
  *a = (int *) malloc(sizeof(int) * 2);
  (*a)[0] = 0;
  (*a)[1] = 1;
  return **a;
}

int main()
{
  int res;
  int p;
  int *q;
  q = &p;
  res = foo(&q);
  return res;
}
