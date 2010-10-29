#include <stdio.h>
#if 0
typedef unsigned long long ticks;

static __inline__ ticks getticks(void)
{
    ticks val;
    __asm__ __volatile__("rdtsc" : "=A" (val));

    return val;
}
#else
#include <time.h>
typedef clock_t ticks;
#define getticks clock
#endif

#define START_PROFILING {\
    ticks start,stop;\
    start=getticks();

#define END_PROFILING \
    stop=getticks();\
    printf("%llu\n",(unsigned long long)(stop-start));\
}
