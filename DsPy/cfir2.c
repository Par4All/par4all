/* cfir2.c - FIR filter implemented with circular delay-line buffer */
void wrap2();

double cfir2(int M, double h[M], double w[M], int *q, double x)
{                        
       int i;
       double y;

       w[*q] = x;                                /* read input sample \(x\) */

       for (y=0, i=0; i<=M; i++) {               /* compute output sample \(y\) */
              y += (*h++) * w[(*q)++];
              wrap2(M, q);
              }

       (*q)--;                                   /* update circular delay line */
       wrap2(M, q);
       
       return y;
}

#include <stdlib.h>
#include <stdio.h>
#include "tools.h"
int main(int argc, char **argv) {
	int n = argc>1 ? atoi(argv[1]) : 1000;
	double r;
	int q=4;
	double (*in)[n] = (double (*)[n])malloc(sizeof(double)*n);
	double (*in2)[n] = (double (*)[n])malloc(sizeof(double)*n);
	init_args(argc, argv);
	init_data_double(*in, n);
	init_data_double(*in, n);
	r=cfir2(n, *in, *in2, &q, 1.4f);
	printf("%ld\n",r);
	return 0;
}
