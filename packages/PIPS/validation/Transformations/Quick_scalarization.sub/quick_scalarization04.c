// two different scalarizable references to the same array
// but in different loop nests -> scalarize

#include <stdio.h>
int main()
{
  int a[2], b[10], i;

  for(i = 0; i< 10; i++)
    {
      a[0] = i;
      b[i] = a[0];
    }

  for(i = 0; i< 10; i++)
    {
      a[1] = i*i;
      b[i] = b[i] + a[1];
    }

  for(i = 0; i< 10; i++)
    {
      printf("%d\n", b[i]);
    }

  return 0;
}
