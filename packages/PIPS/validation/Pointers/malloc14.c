/* Check malloc(NULL) */

#include <malloc.h>

int main()
{
  int * p = NULL;

  free(p);

  return 0;
}
