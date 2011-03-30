/* conv.c - convolution of x[n] with h[n], resulting in y[n] */

#ifdef __PIPS__
#define min MIN
#define max MAX
#else
#define min(a,b) (((a)>(b))?(b):(a))
#define max(a,b) (((a)>(b))?(a):(b))
#endif

void conv(int M, float h[1+M], int L, float x[L+M], float y[L+M])
{
       int n, m;

       for (n = M; n < L+M; n++)
              for (y[n] = 0, m = 0; m < M; m++)
                     y[n] += h[m] * x[n+m];
}
#ifdef CONV_MAIN
#include <stdlib.h>
#include <stdio.h>
#include "tools.h"
int main(int argc, char **argv) {
	int n = atoi(argv[1]);
	int m = atoi(argv[2]);
	float (*in)[n+m] = (float (*)[n+m])malloc(sizeof(float)*(n+m));
	float (*kern)[1+m] = (float (*)[1+m])malloc(sizeof(float)*(1+m));
	float (*out)[n+m] = (float (*)[n+m])malloc(sizeof(float)*(n+m));
	if (argc < 4) {
		fprintf(stderr, "Usage: %s kernel_size conv_size data_file", argv[0]);
		return 1;
	}
	init_data_file(argv[3]);
	init_data_float(*in,n+m);
	init_data_float(*kern,m+1);
	conv(m,*kern,n,*in,*out);
	print_array_float("res",*out,n+m);
	return 0;
}
#endif
