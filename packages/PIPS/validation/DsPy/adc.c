/* adc.c - successive approximation A/D converter */

#include <math.h>

double dac(int B,int b[B],double R);
int u(double);

void adc(double x, int B, int b[B], double R)
{
       int i;
       double y, xQ, Q;

       Q = R / pow(2, B);                        /* quantization width \(Q=R/2\sp{B}\) */
       y = x + Q/2;                              /* rounding */

       for (i = 0; i < B; i++)                   /* initialize bit vector */
              b[i] = 0;

       b[0] = 1 - u(y);                          /* determine MSB */

       for (i = 1; i < B; i++) {                 /* loop starts with \(i=1\) */
              b[i] = 1;                          /* turn \(i\)th bit ON */
              xQ = dac(B, b, R);                 /* compute DAC output */
              b[i] = u(y-xQ);                    /* test and correct bit */
              }
}

#ifdef ADC_MAIN
#include <stdlib.h>
#include <stdio.h>
#include "tools.h"

int main(int argc, char **argv) {
	int n = argc>1 ? atoi(argv[1]) : 1000;
	int (*in)[n] = (int (*)[n])malloc(sizeof(int)*n);
	init_args(argc, argv);
	init_data_int(*in, n);
	adc(4.2,n,*in,3.14);
	print_array_int("r",*in,n);
	return 0;
}
#endif
