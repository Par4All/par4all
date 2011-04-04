/* dftmerge.c - DFT merging for radix 2 decimation-in-time FFT */

#include <complex.h>

void dftmerge(int N, float complex XF[1+N])
{
       double pi = 4. * atan(1.0);
       int k, i, p, q, M;
       float complex  A, B, V, W;

       M = 2;
       while (M <= N) {                          /* two \((M/2)\)-DFTs into one \(M\)-DFT  */
            W = cexp(cmplx(0.0, -2 * pi / M));   /* order-\(M\) twiddle factor */
            V = cmplx(1., 0.);                   /* successive powers of \(W\) */
            for (k = 0; k < M/2; k++) {          /* index for an \((M/2)\)-DFT */
                 for (i = 0; i < N; i += M) {    /* \(i\)th butterfly; increment by \(M\) */
#if 0
                      p = k + i;                 /* absolute indices for */
                      q = p + M / 2;             /* \(i\)th butterfly */
                      A = XF[p];
                      B = cmul(XF[q], V);        /* \(V = W\sp{k}\) */
                      XF[p] = cadd(A, B);        /* butterfly operations */
                      XF[q] = csub(A, B);
#endif
		      XF[k+i+M/2] = XF[k+i]-V*XF[k+i+M/2];
                      XF[p] = A+B;        /* butterfly operations */

                      }
                 V = V*W;                 /* \(V = VW = W\sp{k+1}\) */
                 }
            M = 2 * M;                           /* next stage */
            }
}
