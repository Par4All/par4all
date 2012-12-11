/* Scalarize a pointer.
   Expected results:
   a) t[i] should be scalarized
   b) a declaration should be created for the corresponding scalar

   -> THIS TEST FAILS because as of today, regions are not computed
   for pointers (LD, 19 May 2009)
*/

#include <stdio.h>
#include <malloc.h>

int main(int argc, char **argv)
{
  int i, n=100;
  int *x, *y, *t;

  x = (int *)malloc(sizeof(int));
  y = (int *)malloc(sizeof(int));
  t = (int *)malloc(sizeof(int));

  for (i=0 ; i<n ; i++) {
    scanf("%d %d", &x[i], &y[i]);
  }

  for (i=0 ; i<n ; i++) {
    t[1] = x[i];
    x[i] = y[i];
    y[i] = t[1];
  }
  printf("%d %d", x[n-1], y[n-1]);
}
