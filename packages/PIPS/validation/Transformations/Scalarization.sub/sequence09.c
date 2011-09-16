/* Check that the volatile qualifier on arrays prevents scalarization
 *
 * Issue:
 *
 * Bug: why do I need to separate the sequence from the array
 * declaration? This is an advantage in case an array declaration is
 * put in the middle of a sequence and the initialization of the
 * replacing scalar
 */

#include <stdio.h>
#define SIZE 10

int sequence09()
{
  int k;
  volatile int x[10], y[10][10];
  {
    // to avoid a sequence merge by the controlizer
    int l;
    x[1] = y[1][1];
    x[1] = x[1] + y[1][2];
    x[1] = x[1] + y[1][1];
    x[1] = x[1] + y[1][1];
    k = x[1];
  }

  return k;
}
