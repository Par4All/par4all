/* Check the non detection of dnagling pointers */

#include <malloc.h>

int malloc07(int argc, char *argv[])
{
  int * pi;
  int * qi;

  pi = (int *) malloc(sizeof(int));
  qi = pi;
  free(pi);

  return 0;
}
