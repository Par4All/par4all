#define N 10

int partial_sum(int a[N], int n)
{
  int i;
  int s = 0;
  for (i = 0; i <n; i++)
    {
      s = s+a[i];
    }
  return s;
}

int main()
{
  int i;
  int a[N];
  int k = 1;
  int s = 0;

  for (i = 0; i <10; i++)
    {
      a[i] = k;
    }

  s=partial_sum(a, 5);

  return s;
}
