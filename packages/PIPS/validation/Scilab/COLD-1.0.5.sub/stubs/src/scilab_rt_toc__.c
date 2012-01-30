
#include <stdlib.h>
#include <sys/time.h>

extern double _tictoc;

double scilab_rt_toc__()
{
  return _tictoc + rand();
}


