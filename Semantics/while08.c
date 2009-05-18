/* Side effects in while condition: slight change with respect t0 while08; no implicit condition
 */

#include <stdio.h>

int while08()
{
  int i, n;

  i = 0;
  n = 10;

  while(--n>0) {
    printf("loop: i=%d, n=%d, i+n=%d\n", i, n, i+n);
    i++;
  }

  printf("exit: i=%d, n=%d, i+n=%d\n", i, n, i+n);

  return i;
}

main()
{
  int i;

  i = while08();

  // If the returned value is ignored, PIPS fails:

  // while08();
}
