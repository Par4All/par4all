/* fir.c - FIR filter in direct form */

void fir(int M, double h[M], double w[M], double x, double *r)
{                        
       int i;
       double y;                             /* output sample */

       w[0] = x;                             /* read current input sample \(x\) */

       for (y=0, i=0; i<=M; i++)
              y += h[i] * w[i];              /* compute current output sample \(y\) */

       for (i=M; i>=1; i--)                  /* update states for next call */
              w[i] = w[i-1];                 /* done in reverse order */

       *r = y;
}

#include <stdlib.h>
#include <stdio.h>
#include "tools.h"
int main(int argc, char **argv) {
	int n = argc>1 ? atoi(argv[1]) : 1000;
	float (*in)[n*n] = (float (*) [n*n])malloc(sizeof(float)*n*n);
	float (*in2)[n] = (float (*) [n])malloc(sizeof(float)*n);
	double r;
	init_args(argc, argv);
	init_data_float(*in,n*n);
	fir(n,*in,*in2,0.6f,&r);
	print_array_float("res",&r,1);
	return 0;
}
