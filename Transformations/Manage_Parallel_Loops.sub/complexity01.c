// a costly parallel loop which must be kept parallel
#define max 10000
#include <math.h>
int main()
{
  int a[max];

  for (int i = 0; i<max; i++)
    {
      a[i] = sin(i*1.0);
    }
}
