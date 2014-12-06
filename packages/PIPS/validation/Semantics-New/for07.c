/* Core dump due to typedef statement */

#include <stdlib.h>

void for07(int n)
{
  int a[n], b[n];
  for(int i=0; i<n; i++) {
    a[i] = i;
    typedef int mytype;
    mytype x;
    x = rand();
    b[i] = x;
  }
}
