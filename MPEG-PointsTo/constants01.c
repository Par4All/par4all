// Check handling of C shift operators

// The purpose is to use them to avoid magnitude issue with C code
// including large constants, especially power of 2, in the short term

// In the longer term, the constant expression evaluation should be
// improved to handle these constants

#include <stdio.h>

int main()
{
  int i = 2<<6-1; // i == 64
  int j = 2<<11-1; // j == 2048
  printf("i=%d, j=%d\n", i, j);
  return i+j;
}
