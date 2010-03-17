/* Side effects in while condition
 */

#include <stdio.h>

int while05()
{
  int i, n;

  i = 0;
  n = 10;

  while(--n) {
    printf("loop: i=%d, n=%d, i+n=%d\n", i, n, i+n);
    i++;
  }

  printf("exit: i=%d, n=%d, i+n=%d\n", i, n, i+n);

  return i;
}

main()
{
  int i;

  i = while05();

  // If the returned value is ignored, PIPS fails:

  // while05();
}
