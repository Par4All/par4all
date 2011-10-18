// scalarization of a local array

#include <stdio.h>
int main()
{
  int b[10], i;

  for(i = 0; i< 10; i++)
    {
      int a[10];
      a[i] = i;
      b[i] = a[i];
    }

  for(i = 0; i< 10; i++)
    {
      printf("%d\n", b[i]);
    }

  return 0;
}
