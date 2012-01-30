// Check boolean transformer

// Decomposed version of boolean06.c

#include <stdbool.h>

int main(void)
{
  bool b, t;
  int i = 1, j = 2, k;

  t = i <= j;
  b = b && t;

  return 0;
}
