/* Scalarize an array.
   Expected results:
   a) t[i] scalarized
   b) declaration created for the corresponding scalar
*/

#include <stdio.h>

int main(int argc, char **argv)
{
  int i, n=100;
  int x[n], y[n], t[n];

  for (i=0 ; i<n ; i++) {
    scanf("%d %d", &x[i], &y[i]);
  }

  for (i=0 ; i<n ; i++) {
    t[i] = x[i];
    x[i] = y[i];
    y[i] = t[i];
  }
  printf("%d %d", x[n-1], y[n-1]);
}
