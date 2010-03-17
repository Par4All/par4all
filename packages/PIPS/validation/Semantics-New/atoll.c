/* Make sure that summary preconditions are computed when functions
   are called from within declarations */

#include <stdio.h>
#include <stdlib.h>

long long test_intr (char** argv) {
  long long result = 0;
  result = atoll (argv[1]);
  return result;
}

int main (int argc, char** argv) {
  long long size = test_intr (argv);
  return (int) size;
}
