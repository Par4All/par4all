/* Same as memcopy02b, but the formal parameters src and dst are
 * declared void * instead of char *.
 *
 */

// Definition of NULL
#include <stdio.h>

void memcopy02a(int size, void * src, void * dst)
{
  int i;
  
  if (src!=NULL && dst!=NULL) 
  {
    char* s = (char*) src;
    char* d = (char*) dst;
    for (i=0; i<size; i++)
      d[i] = s[i];
  }
}
