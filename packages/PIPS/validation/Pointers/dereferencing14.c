/* Derived from dereferencin11, example from Serge Guelton.
 *
 * Simplified. Very much.
 *
 * Bug: the first call to foo, foo(zizi) is OK. The second call is
 * identical, but ends up with a typing error. The expression
 * "((fifi+(3-1-0+1)*1)[0])" is not analyzed in the same way when it
 * appears as the right hand side fof the assignment to "zizi" and
 * when it appears as actual argument in the call for foo.
 */

#include<stdio.h>

void foo(int *i)
{
  *i = 999;
}

int duck1(int riri[10], int fifi[4][3], int size, int loulou[20][size][6])
{
  // Here the array "fifi" is cast into an int *, and then an offset is computed
  // fifi+(3-1-0+1)*1 points towards fifi[0][3],
  int *zaza = (int *) fifi+(3-1-0+1)*1;

  // Here an offset is computed first
  // fifi+(3-1-0+1)*1 points towards fifi[3], and the cast makes it a
  // pointer towards fifi[3][0]
  int *zuzu = (int *) (fifi+(3-1-0+1)*1);

  // zizi points to fifi[3][0]
  int *zizi = ((fifi+(3-1-0+1)*1)[0]);

  printf("fifi=%p, zaza=%p, zizi=%p, zuzu=%p\n", fifi, zaza, zizi, zuzu);

  foo(zizi);
  foo((fifi+(3-1-0+1)*1)[0]);
  /* proper effects are not precise here for loulou because 
     of the internal representation of the expression */
  return *((int *) riri+2) = *(zaza+1)+*((int *) loulou+3+(6-1-0+1)*(0+(size-1-0+1)*0));
}
