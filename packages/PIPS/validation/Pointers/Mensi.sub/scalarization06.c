/* Scalarize a pointer.
   Expected results:
   a) t[i] should be scalarized
   b) a declaration should be created for the corresponding scalar

   -> THIS TEST FAILS because as of today, regions are not computed
   for pointers (LD, 19 May 2009)
*/

#include <stdio.h>
#include <malloc.h>

int main(int *x, int *y, int*t)
{
  int i, n=100;

    
  for (i=0 ; i<n ; i++) {
    t[i] = x[i];
    x[i] = y[i];
    y[i] = t[i];
  }
  
}
