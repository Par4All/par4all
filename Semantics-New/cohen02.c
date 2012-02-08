// Example submitted by Albert Cohen for general induction variable detection and substitution

// The loop either enter and is never exited, or it is skipped

// Restructured version of cohen01

// Note: the internal while loop is recovered by Ronan, i.e. the
// induction variable j is recognized

#include <stdio.h>

void cohen02(int n)
{
  int i = 0;
  int j = 42;
  while (i < n) {
    while(i<n && j>=0)
      j -= 2;
    while(i<n && j<0)
      j = 0;
    /* j==0 here */
    printf("i=%d, j=%d\n", i, j);
  } 
  printf("i=%d, j=%d\n", i, j);
}
