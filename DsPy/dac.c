/* dac.c - bipolar two's complement D/A converter */

double dac(int B, int b[B], double R)
                         /* bits are dimensioned as \(b[0], b[1], \dotsc, b[B-1]\) */

{
       int i;
       double dac = 0;

       b[0] = 1 - b[0];                          /* complement MSB */

       for (i = B-1; i >= 0; i--)                /* H\"orner's rule */
          dac = 0.5 * (dac + b[i]);

       dac = R * (dac - 0.5);                    /* shift and scale */

       b[0] = 1 - b[0];                          /* restore MSB */

       return dac;
}
#ifdef DAC_MAIN
#include <stdlib.h>
#include <stdio.h>
#include "tools.h"
int main(int argc, char **argv) {
	double out;
	int n = argc>1 ? atoi(argv[1]) : 1000;
	int (*in)[n] = (int (*)[n])malloc(sizeof(int)*n);
	init_args(argc, argv);
	init_data_int(*in, n);
	out = dac(n,*in,3.14);
	printf("%ld\n",out);
	return 0;
}
#endif
