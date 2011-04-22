#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
//#include "tools.h"

extern void alphablending(unsigned int n, float src1[n], float src2[n], float result[n], float alpha);

int main(int argc, char **argv)
{
	unsigned int n;
	int r;
	float *src1, *src2, *result;
	float alpha;
	n = atoi(argv[1]);
	printf(">>>> %d <<<<\n", n);
	alpha = 0.7;

#define xmalloc(p)							\
	do {								\
		if (posix_memalign((void **) &p, 32, n * sizeof(float))) \
			return 3;					\
	} while(0);

	xmalloc(src1);
	xmalloc(src2);
	xmalloc(result);

	init_data_file(argv[2]);
	init_data_float(src1, n);
	init_data_float(src2, n);
	close_data_file();

	alphablending(n, src1, src2, result, alpha);

	print_array_float("result", result, n);

	free(src1);
	free(src2);

	r = (int) (result[0] + result[n - 1]);
	free(result);

	return 0;
}

void alphablending(unsigned int n, float src1[n], float src2[n], float result[n], float alpha)
{
    unsigned int i,j;
    for(i=0;i<n;i++)
	result[i]=alpha*src1[i]+(1-alpha)*src2[i];
}

