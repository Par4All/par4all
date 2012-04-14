/* Check detection of memory leaks */

#include <malloc.h>

int main(int argc, char *argv[])
{
  int * pi;
  int * qi;
  int i;

  pi = (int *) malloc(sizeof(int));
  if(i>0)
    qi = pi;
  else
    qi = &i;
  free(pi);

  return 0;
}
