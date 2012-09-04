/* Check the use of pointer syntax with an array
 *
 * No points-to information should be generated for this piece of code.
 *
 * Copy of function foo2 in array03.c
 */

#include <stdlib.h>
#include <stdio.h>

#define N 5
#define M 3

int array14(float b[N][M])
{
  float c;
  (*b)[3] = 2.0;
  c = (*b)[3];
  b[1][3] = 2.0;
  c = b[1][3];
  
  ((*b)[3])++;
  (*b)[3] += 5.0;
  (b[1][3])++;
  b[1][3] += 5.0;

  return 1;
}
