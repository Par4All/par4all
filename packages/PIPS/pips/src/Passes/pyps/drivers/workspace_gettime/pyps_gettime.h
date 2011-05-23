#ifndef PYPS_GETTIME_H
#define PYPS_GETTIME_H

#include <sys/time.h>

void __pyps_bench_start(struct timeval *timestart);
void __pyps_bench_stop(const char* module, const struct timeval *timestart);

#endif //PYPS_GETTIME_H
