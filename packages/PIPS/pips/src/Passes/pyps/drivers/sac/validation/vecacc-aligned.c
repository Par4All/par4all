#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
//#include "tools.h"

void vecacc(int n, float* src1, float* src2, float* result)
{
    int i;
#pragma vector aligned
#pragma ivdep
    for(i=0;i<n;i++)
	result[i]=src1[i]*src2[i];
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
	do {\
		p=(float*)malloc(n*sizeof(float));\
	} while(0)

	xmalloc(src1);
	xmalloc(src2);
	xmalloc(result);

	init_data_file(argv[2]);
	init_data_float(src1, n);
	init_data_float(src2, n);
	close_data_file();

	vecacc(n, src1, src2, result);

	print_array_float("result", result, n);

	free(src1);
	free(src2);
	free(result);

	return 0;
}


