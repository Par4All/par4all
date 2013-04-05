/* Same as memcopy05b, but the formal parameters src and dst are
 * declared void * instead of char *.
 *
 */

// Definition of NULL
#include <stdio.h>

void memcopy05a(int size, void* src, void* dst)
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
