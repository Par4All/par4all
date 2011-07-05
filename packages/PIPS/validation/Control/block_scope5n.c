/* Jump into a block from a goto before the block
 *
 * The internal declaration "int x=7;" is badly placed and the
 * generated code does not compile.
 *
 * It is not clear if it is a controlizer issue or a prettyprinter
 * issue. It looks like a prettyprinter issue because the "x--;" is
 * moved up because of the "goto lab1;". However, the controlizer is
 * supposed to move the declaration upwards...
 *
 * Output:
 *
 * internal x = -1 (or 4195391...)
 * external x = 7
 * internal x = 6
 * external x = 3
 *
 * This shows that the internal x is not initialized when the
 * initialization is skipped.
 */

#include <stdio.h>

//#define x1 x
//#define x2 x

void block_scope5n()
{
  int x1 = 6;
  goto lab1;
 lab2:
  x1 = 2;
  {
    int x2 = 7;
  lab1:
    x2--;
    printf("internal x = %d\n", x2);
  }
  x1++;
  printf("external x = %d\n", x1);
  if(x1>3) goto lab2;
}

main()
{
  block_scope5n();
  return 0;
}
