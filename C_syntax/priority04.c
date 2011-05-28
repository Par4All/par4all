#include <stdio.h>


int main() {
  // The precedence is for the unary and then the cast and finally the division
  printf("Should output the same result : %d %d\n", (unsigned char) - 1 / 255, ((unsigned char) - 1) / 255);
  return 0;
}

