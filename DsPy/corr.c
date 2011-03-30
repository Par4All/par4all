/* corr.c - sample cross correlation of two length-N signals */
#include "tools.h"

void corr(int N, float x[1+N], float y[1+N], int M, float R[M])                  /* computes \(R[k]\), \(k = 0, 1,\dotsc, M\) */
                        /* \(x,y\) are \(N\)-dimensional */
                        /* \(R\) is \((M+1)\)-dimensional */
{
       int k, n;

       for (k=0; k<=M; k++)
              for (R[k]=0, n=0; n<N-k; n++)
                     R[k] += x[n+k] * y[n] / N;
}

#include <stdlib.h>
#include <stdio.h>
#include "helper.h"
int main(int argc, char **argv) {
	int n = atoi(argv[1]);
	float (*in)[1+n] = (float (*)[1+n])malloc(sizeof(float)*(1+n));
	float (*in2)[1+n] = (float (*)[1+n])malloc(sizeof(float)*(1+n));
	float (*out)[1+n] = (float (*)[1+n])malloc(sizeof(float)*(1+n));
	init_args(argc, argv);
	init_data_float(*in,1+n);
	init_data_float(*in2,1+n);
	corr(n,*in,*in2,n,*out);
	print_array_float("out",*out,n+1);
	return 0;
}
