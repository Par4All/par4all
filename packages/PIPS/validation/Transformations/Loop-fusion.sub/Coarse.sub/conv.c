/* conv.c - convolution of x[n] with h[n], resulting in y[n] */

#ifdef __PIPS__
#define min MIN
#define max MAX
#else
#define min(a,b) (((a)>(b))?(b):(a))
#define max(a,b) (((a)>(b))?(a):(b))
#endif
#include <stddef.h>

void conv(int M, float h[1+M], int L, float x[L+M], float y[L+M])
                        /* \(h,x,y\) = filter, input, output arrays */
                        /* \(M\) = filter order, \(L\) = input length */
{
       int n, m;

       for (n = M; n < L+M; n++)
              for (y[n] = 0, m = 0; m < M; m++)
                     y[n] += h[m] * x[n+m];
}

#ifdef CONV_MAIN
#include <stdlib.h>
#include <stdio.h>
#include "helper.h"
int main(int argc, char **argv) {
	int n = argc>1 ? atoi(argv[1]) : 1000;
	int m = argc>2 ? atoi(argv[2]) : 5;
	if(n>1) {
		float (*in)[n+m] = (float (*)[n+m])malloc(sizeof(float)*(n+m));
		float (*kern)[1+m] = (float (*)[1+m])malloc(sizeof(float)*(1+m));
		float (*out)[n+m] = (float (*)[n+m])malloc(sizeof(float)*(n+m));
		finit(n+m,*in);
		finit(1+m,*kern);
		conv(m,*kern,n,*in,*out);
		fshow(n+m,*out);
	}
	return 0;
}
#endif
