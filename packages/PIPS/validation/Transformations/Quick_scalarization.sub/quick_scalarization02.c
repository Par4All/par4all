// nested loops case

#include <stdio.h>
int main()
{
  int a[10], b[10][10], i, j;

  for(i = 0; i< 10; i++)
    {
      a[i] = i;
      for(j = 0; j< 10; j++)
	b[i][j] = a[i];
    }

  for(i = 0; i< 10; i++)
    for(j = 0; j< 10; j++)
      {
	printf("%d\n", b[i][j]);
      }

  return 0;
}
