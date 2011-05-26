#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include "tools.h"

void vecacc(int n, float* src1, float* src2, float* result)
{
    int i;
/*#pragma vector aligned
#pragma ivdep*/
    for(i=0;i<n;i++)
	result[i]+=src1[i]*src2[i];
}

int main(int argc, char **argv)
{
	int n;
	float *src1, *src2, *result;
	n = atoi(argv[1]);

#define xmalloc(p)							\
	do {								\
		if (posix_memalign((void **) &p, 32, n * sizeof(float))) \
			return 3;					\
	} while(0);

#define vmalloc(p)\
	do {								\
		p=(float*)malloc(n*sizeof(float));\
	} while(0);

	vmalloc(src1);
	vmalloc(src2);
	vmalloc(result);

	init_data_file(argv[2]);
	init_data_float(src1, n);
	init_data_float(src2, n);
	close_data_file();

	//struct timeval start,end;
	//gettimeofday(&start, NULL);
	struct timespec start,end;
	clock_gettime(CLOCK_MONOTONIC, &start);
	vecacc(n, src1, src2, result);
	clock_gettime(CLOCK_MONOTONIC, &end);
	double tdiff = end.tv_sec*1000.0+(end.tv_nsec/1000000.0) - (start.tv_sec*1000.0+(start.tv_nsec/1000000.0));
	printf("Time: %f\n", tdiff);

	free(src1);
	free(src2);
	free(result);

	return 0;
}


