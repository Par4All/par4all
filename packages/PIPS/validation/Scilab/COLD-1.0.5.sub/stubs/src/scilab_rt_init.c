
#include <stdio.h>
#include <sys/time.h>

#define   SCILAB_NAN      0
#define   SCILAB_INF      1
#define __HUGE_VAL_MAX__  1
#define __HUGE_VAL_MIN__  0
#define __INT_VAL_MAX__   INT_MAX
#define __INT_VAL_MIN__   INT_MIN

const int __SCILAB_RT_FALSE__ = 0;
const int __SCILAB_RT_TRUE__  = 1;

const double SCILAB_E   = 2.7182818284590452354;
const double SCILAB_PI  = 3.14159265358979323846;
const double SCILAB_EPS = 2.22045e-16;

int __scilab_exit__     = 0;
int __scilab_verbose__  = 0;

int __scilab_is_running__ = 0;

double _tictoc = 0.0;

void scilab_rt_init(int argc, char* argv[], int using_jni)
{
  // Not to be executable since it is a stub...
  printf("Hello");
}

