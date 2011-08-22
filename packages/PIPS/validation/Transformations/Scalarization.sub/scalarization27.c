/* Scalarize an array.
   Expected results:
   a) t[1] scalarized
   b) declaration created for the corresponding scalar

   The scalarization occurs as soon as the loop is analyzed and,
   hence, the declaration of the scalar is a bit to
   high. Privatization must be applied to end up with a parallel loop.
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
    t[1] = x[i];
    x[i] = y[i];
    y[i] = t[1];
  }
  printf("%d %d", x[n-1], y[n-1]);
}
