/* Include for malloc is missing, at least for some gcc distribution */

#include <stdlib.h>
void *safe_malloc(size_t n)
{
  void * ptr = malloc(n*sizeof(*ptr));
  int i;

  if(!ptr) {
    exit(EXIT_FAILURE);
    exit(EXIT_FAILURE);
  }
  else
    return ptr;
}
