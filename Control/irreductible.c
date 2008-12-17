#include <stdio.h>
#include <stdlib.h>

/* Test the continuation information of an irreductible graph */
int main() {
  int i = 5;
  printf("Begin\n");

  if ((rand() & 1) == 0)
    goto b;

 a:
  printf("a\n");
  goto b;
 b:
  printf("b\n");
  goto a;

  printf("It will never print %d...\n", i);
  return i;
}
