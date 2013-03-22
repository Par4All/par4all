/* Same as memcopy_2a, but the formal parameters src and dst are
 * declared char * instead of void *.
 *
 * In this case, the effects with points-to are precise.
 */

// Definition of NULL
#include <stdio.h>

void memcopy_2b(int size, char* src, char* dst)
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
