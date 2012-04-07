#include <stdio.h>


int modulo07(int argc, char **argv) {
  unsigned int n=atoi(argv[1]);
  int j = n%4;
  
  // n is unsigned, j cannot be < 0
  return j;
}


