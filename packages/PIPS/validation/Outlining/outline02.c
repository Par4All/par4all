#include <stdio.h>

int main(int argc, char *argv[]) {
  int i, j;

 init_kernel:
  i = 1;
 compute_kernel:
  j = i + 2;

  printf("Value is %d\n", j);

  return 0;
}
