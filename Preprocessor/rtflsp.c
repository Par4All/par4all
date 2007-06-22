#include <math.h>

#define MAXIT 30

float rtflsp(func,x1,x2,xacc)
float x1,x2,xacc;
float (*func)();	/* ANSI: float (*func)(float); */
{
	int j;
	float fl,fh,xl,xh,swap,dx,del,f,rtf;
	void nrerror();

	fl=(*func)(x1);
	fh=(*func)(x2);
	if (fl*fh > 0.0) nrerror("Root must be bracketed in RTFLSP");
	if (fl < 0.0) {
		xl=x1;
		xh=x2;
	} else {
		xl=x2;
		xh=x1;
		swap=fl;
		fl=fh;
		fh=swap;
	}
	dx=xh-xl;
	for (j=1;j<=MAXIT;j++) {
		rtf=xl+dx*fl/(fl-fh);
		f=(*func)(rtf);
		if (f < 0.0) {
			del=xl-rtf;
			xl=rtf;
			fl=f;
		} else {
			del=xh-rtf;
			xh=rtf;
			fh=f;
		}
		dx=xh-xl;
		if (fabs(del) < xacc || f == 0.0) return rtf;
	}
	nrerror("Maximum number of iterations exceeded in RTFLSP");
}

#undef MAXIT
