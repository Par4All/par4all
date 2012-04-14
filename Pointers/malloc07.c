/* Check detection of memory leaks */

#include <malloc.h>

int main(int argc, char *argv[])
{
  int * pi;
  int * qi;

  pi = (int *) malloc(sizeof(int));
  qi = pi;
  free(pi);

  return 0;
}
