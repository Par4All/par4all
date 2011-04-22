#ifndef PYPS_TIME_FILE
	#error pyps should define PYPS_TIME_FILE on its own
#endif
#include <sys/time.h>
#include <stdio.h>

// Warning: these functions aren't thread-safe !!

static FILE *__pyps_timefile = 0;

void __pyps_bench_start(struct timeval *timestart)
{
	gettimeofday(timestart, NULL);
}

void __pyps_bench_stop(const char* module, const struct timeval *timestart)
{
	struct timeval timeend;
	gettimeofday(&timeend, NULL);
	
	long diff = (timeend.tv_sec-timestart->tv_sec)*1000000 + (timeend.tv_usec-timestart->tv_usec);
	
	if (__pyps_timefile == 0)
		__pyps_timefile = fopen(PYPS_TIME_FILE, "w");
	if (__pyps_timefile)
	{
		fprintf(__pyps_timefile, "%s: %ld\n", module, diff);
		fflush(__pyps_timefile);
	}
}

void __pyps_bench_close(void)
{
	if (__pyps_timefile != 0)
		fclose(__pyps_timefile);
}
atexit(__pyps_bench_close);
