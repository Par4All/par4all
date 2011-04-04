/* modwrap.c - modulo-N wrapping of length-L signal */

void modwrap(int L, float x[L], int N, float xtilde[N])                /* usage: modwrap(L, x, N, xtilde); */
                          /* \(x\) is \(L\)-dimensional */
                          /* xtilde is \(N\)-dimensional */
{
    int n, r, m, M;

    r = L % N;                               /* remainder \(r=0,1,\dotsc,N-1\) */
    M = (L-r) / N;                           /* quotient of division \(L/N\) */

    for (n=0; n<N; n++) {
         if (n < r)                          /* non-zero part of last block */
             xtilde[n] = x[M*N+n];           /* if \(L<N\), this is the only block */
         else
             xtilde[n] = 0;                  /* if \(L<N\), pad \(N-L\) zeros at end */

         for (m=M-1; m>=0; m--)              /* remaining blocks */
              xtilde[n] += x[m*N+n];         /* if \(L<N\), this loop is skipped */
         }
}

#include <stdlib.h>
#include <stdio.h>
#include "tools.h"
int main(int argc, char **argv) {
	int n = argc>1 ? atoi(argv[1]) : 1000;
	float (*in)[n*n] = (float (*) [n*n])malloc(sizeof(float)*n*n);
	float (*out)[n] = (float (*) [n])malloc(sizeof(float)*n);
	init_args(argc, argv);
	init_data_float(*in,n*n);
	modwrap(n*n,*in,n,*out);
	print_array_float("res",*out,n);
	return 0;
}
