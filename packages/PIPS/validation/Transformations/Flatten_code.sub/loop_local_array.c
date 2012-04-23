// It is not legal to move declaration of array b outside the external loop
// after it has been declared as parallel
// Hence array b must either be kept local, or the surrounding loop be declared
// as sequential (current status of flatten code (2012-04-19))
#include <stdio.h>
int main()
{
  int i,j, a[10][20];
  for (i=0; i< 10; i++)
    {
      int b[1][20];
      for(j=0; j<20; j++)
	b[0][j] = j;

      for (j=0; j<20; j++)
	a[i][j] = b[0][j];
    }

    for (i=0; i< 10; i++)
    {
      for(j=0; j<20; j++)
	printf("a[%d][%d] = %d\n", i, j, a[i][j]);
    }
  return 0;
}
