/* Check detection of memory leaks */

#include <malloc.h>

int main(int argc, char *argv[])
{
  int * pi;

  pi = (int *) malloc(sizeof(int));

  free(pi);

  return 0;
}
