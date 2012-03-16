# include <stdlib.h>

int foo(int *a[10])
{
  int failed = 0;
  for (int i = 0; i<10; i++)
    {
      a[i] = (int *) malloc (5 * sizeof(int));
      if (a[i] == NULL) failed++;
    }
  return failed;
}

int main()
{
  int failed;
  int *a[10];

  failed = foo(a);
  return failed;
}
