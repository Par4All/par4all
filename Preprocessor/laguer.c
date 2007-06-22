#include <math.h>
#include "complex.h"

#define EPSS 6.e-8
#define MAXIT 100

void laguer(a,m,x,eps,polish)
fcomplex a[],*x;
int m,polish;
float eps;
{
	int j,iter;
	float err,dxold,cdx,abx;
	fcomplex sq,h,gp,gm,g2,g,b,d,dx,f,x1;
	void nrerror();

	dxold=Cabs(*x);
	for (iter=1;iter<=MAXIT;iter++) {
		b=a[m];
		err=Cabs(b);
		d=f=Complex(0.0,0.0);
		abx=Cabs(*x);
		for (j=m-1;j>=0;j--) {
			f=Cadd(Cmul(*x,f),d);
			d=Cadd(Cmul(*x,d),b);
			b=Cadd(Cmul(*x,b),a[j]);
			err=Cabs(b)+abx*err;
		}
		err *= EPSS;
		if (Cabs(b) <= err) return;
		g=Cdiv(d,b);
		g2=Cmul(g,g);
		h=Csub(g2,RCmul(2.0,Cdiv(f,b)));
		sq=Csqrt(RCmul((float) (m-1),Csub(RCmul((float) m,h),g2)));
		gp=Cadd(g,sq);
		gm=Csub(g,sq);
		if (Cabs(gp) < Cabs(gm))gp=gm;
		dx=Cdiv(Complex((float) m,0.0),gp);
		x1=Csub(*x,dx);
		if (x->r == x1.r && x->i == x1.i) return;
		*x=x1;
		cdx=Cabs(dx);
		if (iter > 6 && cdx >= dxold) return;
		dxold=cdx;
		if (!polish)
			if (cdx <= eps*Cabs(*x)) return;
	}
	nrerror("Too many iterations in routine LAGUER");
}

#undef EPSS
#undef MAXIT
