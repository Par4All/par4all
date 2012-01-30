
#include <string.h>
#include <stdlib.h>
#include <stdio.h>



char* scilab_rt_ascii_i2_(int n, int m, int res[n][m])
{
  char* s = (char*)malloc(m*sizeof(char)+1);
  for (int i=0; i<m; i++)
    s[i] = (char) res[0][i];;
  s[m] = 0;
  return s;
}

