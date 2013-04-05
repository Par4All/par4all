
// Definition of NULL
#include <stdio.h>

void memcopy04(int size, char* src, char* dst)
{
  int i;
  if (src!=NULL && dst!=NULL) 
  {
    char* s = (char*) src;
    char* d = (char*) dst;
    for (i=0; i<size; i++)
      *d++ = *s++;
  }
}
