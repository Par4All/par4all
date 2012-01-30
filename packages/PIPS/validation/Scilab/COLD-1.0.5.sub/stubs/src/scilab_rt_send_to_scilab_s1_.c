
#include <stdio.h>

void scilab_rt_send_to_scilab_s1_(int n, char* s[n])
{
  int i;

  printf("%d",n);

  for (i = 0; i < n; ++i) {
    printf("%s",s[i]);
  }

}


