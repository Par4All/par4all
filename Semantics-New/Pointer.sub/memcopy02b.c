/* Same as memcopy02a, but the formal parameters src and dst are
 * declared char * instead of void *.
 *
 */

// Definition of NULL
#include <stdio.h>

void memcopy02b(int size, char* src, char* dst)
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
