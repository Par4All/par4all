
#include <stdlib.h>
#include <sys/time.h>

extern double _tictoc;

void scilab_rt_tic__()
{
  _tictoc = rand();
}

