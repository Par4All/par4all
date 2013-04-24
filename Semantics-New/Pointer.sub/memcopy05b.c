/* Same as memcopy05a, but the formal parameters src and dst are
 * declared char * instead of void *.
 *
 */

// Definition of NULL
#include <stdio.h>

void memcopy05b(int size, char* src, char* dst)
{
  if (src!=NULL && dst!=NULL) 
  {
    char* s = (char*) src;
    char* d = (char*) dst;
    char* ls = s+size;
    while (s<ls)
      *d++ = *s++;
  }
}
