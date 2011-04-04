/* dtftr.c - N DTFT values over frequency range [wa, wb) */

#include <complex.h>                                 

void dtftr(int L, float x[L], int N, float complex X[N], float wa, float wb) 
{
       int k;
       float dw = (wb-wa)/N;                      /* frequency bin width */

       for (k=0; k<N; k++)
             dtft(L, x, wa + k*dw, &X[k]);        /* \(k\)th DTFT value \(X(\om\sb{k})\) */
}

#ifdef DTFTR_MAIN
#include <stdlib.h>
#include <stdio.h>
#include "helper.h"
int main(int argc, char **argv) {
	int n = argc>1 ? atoi(argv[1]) : 1000;
	if(n>1) {
		float (*in)[n] = (float (*)[n])malloc(sizeof(float)*(n));
		float complex (*out)[n] = (float complex (*)[n])malloc(sizeof(float complex)*(n));
		finit(n,*in);
		dtftr(n,*in,n,*out,3.14,4.2);
		cshow(n,*out);
	}
	return 0;
}
#endif
