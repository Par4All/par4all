/* Check the impact of property POINTS_TO_SUCCESSFUL_MALLOC_ASSUMED
   when it is set to false. */

#include <malloc.h>

char * malloc21(unsigned int n)
{
  char * p = malloc(n);

  return p;
}
