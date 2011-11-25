/* Check  for constant arguments of modulo */

#include <stdio.h>

int main()
{
  int r1, r2, r3, r4;
  //int a=3, b=2;
  int a=3, b=2, c=0;

  r1 = a % b;
  r2 = -a % b;
  r3 = a % -b;
  r4 = -a % -b;

  // Expected result: r1=1, r2=-1, r3=1, r4=-1
  printf("r1=%d, r2=%d, r3=%d, r4=%d\n", r1, r2, r3, r4);

  r1 = c % b;
  // This causes a zero divide
  r2 = a % c;

  // Expected result: r1=1, r2=-1
  printf("r1=%d, r2=%d\n", r1, r2);

  return 0;
}
