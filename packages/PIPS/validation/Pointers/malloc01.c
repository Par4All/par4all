/* Check the OUT information. */

#include <malloc.h>

char * malloc01(unsigned int n)
{
  char * p = malloc(n);

  return p;
  int i; // added to see the impact of return
}
