#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "stars-pm-cuda.h"

// Timing
double get_time()
{
    struct timeval t;
    struct timezone tzp;
    gettimeofday(&t, &tzp);
    return t.tv_sec + t.tv_usec*1e-6;
}


void do_histo(int *data,
              int *histo) {
  int i;
  memset(histo, 0, NPART * sizeof(int));
  for (i = 0; i < NPART; i++) {
    ++histo[data[i]];
  }
}

void histogram(int data[NP][NP][NP],
               int histo[NP][NP][NP]) {

#ifdef P4A_TIMING
  double start_time = get_time();
#endif

  // Do histogramming
  do_histo((int *)data,(int  *)histo);


#ifdef P4A_TIMING
  double end_time = get_time();
  fprintf(stderr," P4A: Time for '%s' : %fms\n",__FUNCTION__, (end_time-start_time)*1000);
#endif
}


