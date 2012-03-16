int foo(int *a[10])
{
  return *(a[5]);
}

int main()
{
  int res;
  int *a[10];
  for(int i = 0; i<10; i++)
    {
      a[i] = (int *) malloc(sizeof(int));
      *a[i] = i;
    }
  res = foo(a);
  return res;
}
