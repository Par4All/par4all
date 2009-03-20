#include <stdio.h>
#include <stdlib.h>

/* Test the continuation information of a goto loop */
int main() {
  int i = 5;
  printf("Begin\n");

  /* The a label */
 a:
  printf("a\n");
  /* The b label */
 b:
  printf("b\n");
  /* Oh, go to a... */
  goto a;

  /* Unreachable... */
  printf("It will never print %d...\n", i);
  return i;
}
