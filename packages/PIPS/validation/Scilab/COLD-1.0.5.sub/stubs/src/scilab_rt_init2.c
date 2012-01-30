
#include <stdio.h>

extern void code();

void scilab_rt_init2(int argc, char* argv[], int mode, void (* foo )() )
{
  printf("%d", mode);
  code();
}

