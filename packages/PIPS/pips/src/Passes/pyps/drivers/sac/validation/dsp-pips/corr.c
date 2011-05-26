/* corr.c - sample cross correlation of two length-N signals */

#include "tools.h"

void corr(int N, float x[N], float y[N], int M, float R[M])                  /* computes \(R[k]\), \(k = 0, 1,\dotsc, M\) */
                        /* \(x,y\) are \(N\)-dimensional */
                        /* \(R\) is \((M+1)\)-dimensional */
{
       int k, n;

       for (k=0; k<M; k++)
              for (R[k]=0, n=0; n<N-k; n++)
                     R[k] += x[n+k] * y[n] / N;
}

#include <stdlib.h>
#include <stdio.h>
#include "tools.h"
int main(int argc, char **argv) {
	int n = atoi(argv[1]);
	int k = 5;
	//int k = atoi(argv[3]);
	float (*in)[n] = (float (*)[n])malloc(sizeof(float)*(1+n));
	float (*in2)[n] = (float (*)[n])malloc(sizeof(float)*(1+n));
	float (*out)[k] = (float (*)[n])malloc(sizeof(float)*(k));
	init_args(argc, argv);
	init_data_float(*in,n);
	init_data_float(*in2,n);

	if (argc>20) k=10;
	struct timeval s,e;
	gettimeofday(&s,NULL);
	corr(n,*in,*in2,k,*out);
	gettimeofday(&e,NULL);
	double diff = (double)(e.tv_sec-s.tv_sec)*1000.0 +
		(double)(e.tv_usec-s.tv_usec)/1000.0;
	printf("%0.6f\n", diff);
	return 0;
}
