#ifndef P4A_CROUGH_TYPES_H
#define P4A_CROUGH_TYPES_H

#include "stdint.h"

#ifdef P4A_USE_INT64
typedef int64_t p4a_int;
#else
typedef int32_t p4a_int;
#endif

#ifdef P4A_USE_REAL64
typedef double p4a_real;
#else
typedef float p4a_real;
#endif

#endif
