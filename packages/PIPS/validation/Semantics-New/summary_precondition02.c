/* Make sure that summary preconditions are computed when functions
   are called from within declarations.

   Problem with PIPS preprocessor
 */

/*
#include <stdio.h>
#include <stdlib.h>
*/

long long atoll(char *);

long long test_intr (char** argv) {
  long long result = 0;
  result = atoll (argv[1]);
  return result;
}

int main (int argc, char** argv) {
  long long size[test_intr (argv)];
  return (int) size[0];
}
