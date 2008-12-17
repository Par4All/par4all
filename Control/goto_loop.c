#include <stdio.h>
#include <stdlib.h>

/* Test the continuation information of a goto loop */
int main() {
  int i = 5;
  printf("Begin\n");

 a:
  printf("a\n");
  goto b;
 b:
  printf("b\n");
  goto a;

  printf("It will never print %d...\n", i);
  return i;
}
