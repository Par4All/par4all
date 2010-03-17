#include <stdio.h>
#include <stdlib.h>

/* Test the continuation information of an irreductible graph */
int main() {
  int i = 5;
  printf("Begin\n");

  if ((rand() & 1) == 0)
    goto b;

   /* The a label */
a:
  printf("a\n");
  /* The b label */
  goto b;
  /* The b label */
 b:
  printf("b\n");
  /* Oh, go to a... */
 goto a;

  /* Unreachable... */
  printf("It will never print %d...\n", i);
  return i;
}
