#include <stdlib.h>

void for06(int n)
{
  int a[n], b[n];
  for(int i=0; i<n; i++) {
    a[i] = i;
    int x;
    x = rand();
    b[i] = x;
  }
}
