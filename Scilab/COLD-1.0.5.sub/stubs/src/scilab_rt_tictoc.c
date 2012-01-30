
#include<stdlib.h>

static double *scilab_rt_tic__anywhere;

void scilab_rt_tic__()
{
  *scilab_rt_tic__anywhere += rand();
}


double scilab_rt_toc__()
{
  return *scilab_rt_tic__anywhere -= rand();
}
