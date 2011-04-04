/* dtft.c - DTFT of length-L signal at a single frequency w */
#include <complex.h>  
/* complex arithmetic */

void dtft(int L, float x[L], float w, float complex* y) /* usage: X=dtft(L, x, w); */
{
	float complex z, X;
	int n;
    z = cexpf( - I* w);                     /* set \(z=e\sp{-j\om}\) */
    X = 0.f;                               /* initialize \(X=0\) */

	for (n=L-1; n>=0; n--)
	    X =  x[n] + z*X;

	*y =  X;
}
