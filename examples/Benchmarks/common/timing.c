#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <time.h>
#include <sys/time.h>

#include "timing.h"

/* Timer code (gettimeofday). */
static double t_start, t_end;

double get_time()
{
    struct timeval t;
    if (gettimeofday (&t, NULL) != 0) {
      perror("Error gettimeofday !\n");
      exit(1);
    }
    return (t.tv_sec + t.tv_usec * 1.0e-6);
}


void timer_start() {
  t_start = get_time();
}

void timer_stop() {
  t_end = get_time();
}

void timer_display() {
  printf ("%0.1lf\n", (t_end - t_start)*1000);
}

void timer_stop_display() {
  timer_stop();
  timer_display();
}
