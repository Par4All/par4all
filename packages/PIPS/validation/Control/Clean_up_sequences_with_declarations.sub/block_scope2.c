/* Yet another case of ambiguity to resolve
 *
 * OK, when the three variables are different
 *
 * NOK, when they have the same name because the renaming occurs
 * before the initialization is removed. The bug has been fixed, but
 * the semantics is different when the variables are renamed because
 * of the test x1>0.
 */

#define x1 x
#define x2 x
#define x3 x

#include <stdio.h>

int block_scope2()
{
  int x1 = 6;
  {
  lab1:
    x1--;
    {
      int x2;
      x2++;
    }
  }
  {
    int x3 = -1;
    if(x1>0) goto lab1;
  }
  return x1;
}

main()
{
  int j = block_scope2();
  printf("j=%d\n", j);
  return 0;
}
