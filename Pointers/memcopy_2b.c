#include <stdio.h>


void memcopy_2b(int size, char* src, char* dst)
{
  //char* s = (char*) src;
  //char* d = (char*) dst;
  int i;
  
  if (src!=NULL && dst!=NULL) 
  {
    char* s = (char*) src;
    char* d = (char*) dst;
    for (i=0; i<size; i++)
      d[i] = s[i];
  }
}
