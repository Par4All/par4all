/* Same as memcopy_2b, but the formal parameters src and dst are
 * declared void * instead of char *.
 *
 * In this case, the effects with points-to are imprecise and the loop
 * cannot be parallelized.
 */

// Definition of NULL
#include <stdio.h>

void memcopy_2a(int size, void * src, void * dst)
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
