// a single loop case

#include <stdio.h>
int main()
{
  int a[10], b[10], i;

  for(i = 0; i< 10; i++)
    {
      a[i] = i;
      b[i] = a[i];
    }

  for(i = 0; i< 10; i++)
    {
      printf("%d\n", b[i]);
    }

  return 0;
}
