#include <stdio.h>
#include<stdlib.h>
// parse the command arguments to get the matrix size
long long get_result (int argc, char** argv) {
  long long result = 0;
  result = atol (argv[1]);
  printf ("return value will be %lld\n", result);
  return result;
}

int main (int argc, char** argv) {
  long long size = get_result (argc, argv);
  return size;
}
