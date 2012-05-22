// a simple parallel loop which must NOT be kept parallel
#define max 2
#include <math.h>
int main()
{
  int a[max];

  for (int i = 0; i<max; i++)
    {
      a[i] = sin(i*1.0);
    }
}
