#ifndef __STARS_PM__H_
#define __STARS_PM__H_

#ifdef __cplusplus
extern "C" {
#endif

#include <string.h>
#include "varglob.h"

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
