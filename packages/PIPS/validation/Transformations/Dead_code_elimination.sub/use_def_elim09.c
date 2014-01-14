// Test case added by Francois Irigoin to check dead code elimination
// with out regions
//
// 

#include "stdio.h"

int main(void) {
  int i = 1;
  int j = 2;
  printf("%d\n", i);
  return 0;
}

