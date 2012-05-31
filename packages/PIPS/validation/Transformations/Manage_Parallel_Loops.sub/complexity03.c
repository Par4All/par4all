// a cheap parallel loop on j which must be sequentialized,
// embedded in a costly parallel loop which must be kept parallel
// however, the private clause for variable 'x' of the inner loop
// must be moved to the outer loop
#define max 10000
#include <math.h>
int main()
{
  int a[max][2], x;

  for (int i = 0; i<max; i++)
    {
      for (int j = 0; j<2; j++)
	{
	  x = i +j;
	  a[i][j] = sin(x*1.0);

	}
    }
  return 0;
}
