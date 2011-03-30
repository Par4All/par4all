/* allpass.c - allpass reverberator with circular delay line */
#include <stdlib.h>
#include <stdio.h>
#include "tools.h"

double tap();
void cdelay();

double allpass(int D, double w[], double (*p)[], double a, double x)
{
       double y, s0, sD;

       sD = tap(D, w, *p, D);                   /* \(D\)th tap delay output */
       s0 = x + a * sD;
       y  = -a * s0 + sD;                       /* filter output */
       **p = s0;                                /* delay input */
       cdelay(D, w, p);                         /* update delay line */

       return y;
}

int main(int argc, char **argv) {
	int n = argc>1 ? atoi(argv[1]) : 1000;
	double r;
	double (*in)[n] = (double (*)[n])malloc(sizeof(double)*n);
	double (*in2)[n] = (double (*)[n])malloc(sizeof(double)*n);
	init_args(argc, argv);
	init_data_double(*in, n);
	init_data_double(*in, n);
	r=allpass(4, *in, in2, 0.4f, 0.9f);
	printf("%ld\n",r);
	return 0;
}
