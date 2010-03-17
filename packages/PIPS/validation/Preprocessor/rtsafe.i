#include <math.h>

#define MAXIT 100

float rtsafe(funcd,x1,x2,xacc)
float x1,x2,xacc;
void (*funcd)();	/* ANSI: void (*funcd)(float,float *,float *); */
{
	int j;
	float df,dx,dxold,f,fh,fl;
	float swap,temp,xh,xl,rts;
	void nrerror();

	(*funcd)(x1,&fl,&df);
	(*funcd)(x2,&fh,&df);
	if (fl*fh >= 0.0) nrerror("Root must be bracketed in RTSAFE");
	if (fl < 0.0) {
		xl=x1;
		xh=x2;
	} else {
		xh=x1;
		xl=x2;
		swap=fl;
		fl=fh;
		fh=swap;
	}
	rts=0.5*(x1+x2);
	dxold=fabs(x2-x1);
	dx=dxold;
	(*funcd)(rts,&f,&df);
	for (j=1;j<=MAXIT;j++) {
		if ((((rts-xh)*df-f)*((rts-xl)*df-f) >= 0.0)
			|| (fabs(2.0*f) > fabs(dxold*df))) {
			dxold=dx;
			dx=0.5*(xh-xl);
			rts=xl+dx;
			if (xl == rts) return rts;
		} else {
			dxold=dx;
			dx=f/df;
			temp=rts;
			rts -= dx;
			if (temp == rts) return rts;
		}
		if (fabs(dx) < xacc) return rts;
		(*funcd)(rts,&f,&df);
		if (f < 0.0) {
			xl=rts;
			fl=f;
		} else {
			xh=rts;
			fh=f;
		}
	}
	nrerror("Maximum number of iterations exceeded in RTSAFE");
}

#undef MAXIT
