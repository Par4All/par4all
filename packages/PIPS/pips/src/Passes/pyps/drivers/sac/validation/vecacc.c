#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include "tools.h"

void vecacc(int n, float src1[n], float src2[n], float result[n])
{
    int i;
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

	vecacc(n, src1, src2, result);

	free(src1);
	free(src2);
	free(result);

	return 0;
}


