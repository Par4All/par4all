
#include <stdio.h>

int scilab_rt_mopen_s0s0_(char* filename, char* flags)
{
  printf("%s",filename);
  printf("%s",flags);

  return filename[0];
}

