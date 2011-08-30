#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

#include "timing.h"

#ifndef POLYBENCH_CACHE_SIZE_KB
# define POLYBENCH_CACHE_SIZE_KB 16384
#endif



/**
 *  Always clear the L1/L2/L3 caches before running a benchmark
 *  Imported from Polybench 2.0
 *
 */
void polybench_flush_cache()
{
  int cs = POLYBENCH_CACHE_SIZE_KB * 1024 * 100/ sizeof(double);
  double* flush = (double*) calloc(cs, sizeof(double));
  int i;
  double tmp = 0.0;
// #pragma omp parallel for
  for (i = 0; i < cs; i++)
    tmp += flush[i];
  free(flush);
  assert (tmp <= 10.0);
}



/* Timer code (gettimeofday). */
static double t_start, t_end;

double timer_get_time()
{
    struct timeval t;
    if (gettimeofday (&t, NULL) != 0) {
      perror("Error gettimeofday !\n");
      exit(1);
    }
    return (t.tv_sec + t.tv_usec * 1.0e-6);
}


void timer_start() {
  polybench_flush_cache();
  t_start = timer_get_time();
}

void timer_stop() {
  t_end = timer_get_time();
}

void timer_display() {
  printf ("%0.1lf\n", (t_end - t_start)*1000);
}

void timer_stop_display() {
  timer_stop();
  timer_display();
}


