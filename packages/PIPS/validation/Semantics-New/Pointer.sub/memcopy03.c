
// Definition of NULL
#include <stdio.h>

void memcopy03(int size, void* src, void* dst)
{
  int i=size;
  if (src!=NULL && dst!=NULL) 
  {
    char* s = (char*) src;
    char* d = (char*) dst;
    for (; i>=8; i-=8, s+=8, d+=8)
      *((double*)d) = *((double*)s);
    for (; i!=0; i--, s++, d++)
      *d = *s;
  }
}