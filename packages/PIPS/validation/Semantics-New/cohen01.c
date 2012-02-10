// Example submitted by Albert Cohen for general induction variable detection and substitution

// The loop either enter and is never exited, or it is skipped

#include <stdio.h>

void cohen01(int n)
{
  int i = 0;
  int j = 42;
  while (i < n) {
    j -= 2;
    if (j < 0)
      j = 0;
  } 
  printf("i=%d, j=%d\n", i, j);
}
