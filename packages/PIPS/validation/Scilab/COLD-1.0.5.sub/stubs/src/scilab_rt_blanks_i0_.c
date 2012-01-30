
#include <stdlib.h>

char* scilab_rt_blanks_i0_(int n)
{
  char* res = (char*)malloc(n*sizeof(char)+1);
  for (int i = 0; i<n; i++)
    res[i] = ' ';
  res[n] = 0;
  return res;

}

