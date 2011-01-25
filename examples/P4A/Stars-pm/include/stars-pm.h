#ifndef __STARS_PM__H_
#define __STARS_PM__H_

#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include "varglob.h"

#ifdef CPU_TIMING
#include <stdio.h>
#define TIMING(call) \
{ \
  double end_time,start_time = get_time(); \
  call; \
  end_time = get_time(); \
  fprintf(stderr," P4A: Time for '%s' : %fms\n",#call, (end_time-start_time)*1000); \
}
#else
#define TIMING(call) call
#endif // CPU_TIMING



#if NP == 32
#include "stars-pm-generated_32.h"
#elif NP == 64
#include "stars-pm-generated_64.h"
#elif NP == 128
#include "stars-pm-generated_128.h"
#else
#error NP must be 32, 64 or 128
#endif

#ifdef __cplusplus
}
#endif

#endif // __STARS_PM__H_
