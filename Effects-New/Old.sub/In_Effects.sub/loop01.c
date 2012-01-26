#define N 10

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

  for (i = 0; i <10; i++)
    {
      s = s+a[i];
    }

  return s;
}
