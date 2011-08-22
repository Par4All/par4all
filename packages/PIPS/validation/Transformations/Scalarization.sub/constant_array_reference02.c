// Special case of constant regions, here a reference to a dynamically
// constant array element

#include <stdlib.h>

int main()
{
  int n;
  // This is legal C for gcc, although n is not initialized!
  int a[n],b[n],c[n];
  int i,j, k = n/2;

  // initalize
  for(i=0; i<n; i++) {
    a[i] = i*rand();
  }

  // This loop use only one element of a, it should be scalarized
  for(i=0; i<n; i++) {
    b[i] = rand() * a[k];
  }


  // This is exactly the same loop, but with a surrounding one !
  for(j=0; j<n; j++) {
    for(i=0; i<n; i++) {
      c[i] = rand() * a[k];
    }
  }

  // Use b and c, just for region !
  int x;
  for (i = 1; i <= n; i++) {
    x+= b[i];
    x+= c[i];
  }
  return 0;
}
