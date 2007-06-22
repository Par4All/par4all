#include <math.h>

#define PGROW -0.20
#define PSHRNK -0.25
#define FCOR 0.06666666		/* 1.0/15.0 */
#define SAFETY 0.9
#define ERRCON 6.0e-4

void rkqc(y,dydx,n,x,htry,eps,yscal,hdid,hnext,derivs)
float y[],dydx[],*x,htry,eps,yscal[],*hdid,*hnext;
void (*derivs)();	/* ANSI: void (*derivs)(float,float *,float *); */
int n;
{
	int i;
	float xsav,hh,h,temp,errmax;
	float *dysav,*ysav,*ytemp,*vector();
	void rk4(),nrerror(),free_vector();

	dysav=vector(1,n);
	ysav=vector(1,n);
	ytemp=vector(1,n);
	xsav=(*x);
	for (i=1;i<=n;i++) {
		ysav[i]=y[i];
		dysav[i]=dydx[i];
	}
	h=htry;
	for (;;) {
		hh=0.5*h;
		rk4(ysav,dysav,n,xsav,hh,ytemp,derivs);
		*x=xsav+hh;
		(*derivs)(*x,ytemp,dydx);
		rk4(ytemp,dydx,n,*x,hh,y,derivs);
		*x=xsav+h;
		if (*x == xsav) nrerror("Step size too small in routine RKQC");
		rk4(ysav,dysav,n,xsav,h,ytemp,derivs);
		errmax=0.0;
		for (i=1;i<=n;i++) {
			ytemp[i]=y[i]-ytemp[i];
			temp=fabs(ytemp[i]/yscal[i]);
			if (errmax < temp) errmax=temp;
		}
		errmax /= eps;
		if (errmax <= 1.0) {
			*hdid=h;
			*hnext=(errmax > ERRCON ?
				SAFETY*h*exp(PGROW*log(errmax)) : 4.0*h);
			break;
		}
		h=SAFETY*h*exp(PSHRNK*log(errmax));
	}
	for (i=1;i<=n;i++) y[i] += ytemp[i]*FCOR;
	free_vector(ytemp,1,n);
	free_vector(dysav,1,n);
	free_vector(ysav,1,n);
}

#undef PGROW
#undef PSHRNK
#undef FCOR
#undef SAFETY
#undef ERRCON
