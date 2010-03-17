#include <math.h>

#define MAXSTP 10000
#define TINY 1.0e-30

int kmax=0,kount=0;  /* defining declaration */
float *xp=0,**yp=0,dxsav=0;  /* defining declaration */

void odeint(ystart,nvar,x1,x2,eps,h1,hmin,nok,nbad,derivs,rkqc)
float ystart[],x1,x2,eps,h1,hmin;
int nvar,*nok,*nbad;
void (*derivs)();	/* ANSI: void (*derivs)(float,float *,float *); */
void (*rkqc)(); 	/* ANSI: void (*rkqc)(float *,float *,int,float *,float,
				float,float *,float *,float *,void (*)()); */
{
	int nstp,i;
	float xsav,x,hnext,hdid,h;
	float *yscal,*y,*dydx,*vector();
	void nrerror(),free_vector();

	yscal=vector(1,nvar);
	y=vector(1,nvar);
	dydx=vector(1,nvar);
	x=x1;
	h=(x2 > x1) ? fabs(h1) : -fabs(h1);
	*nok = (*nbad) = kount = 0;
	for (i=1;i<=nvar;i++) y[i]=ystart[i];
	if (kmax > 0) xsav=x-dxsav*2.0;
	for (nstp=1;nstp<=MAXSTP;nstp++) {
		(*derivs)(x,y,dydx);
		for (i=1;i<=nvar;i++)
			yscal[i]=fabs(y[i])+fabs(dydx[i]*h)+TINY;
		if (kmax > 0) {
			if (fabs(x-xsav) > fabs(dxsav)) {
				if (kount < kmax-1) {
					xp[++kount]=x;
					for (i=1;i<=nvar;i++) yp[i][kount]=y[i];
					xsav=x;
				}
			}
		}
		if ((x+h-x2)*(x+h-x1) > 0.0) h=x2-x;
		(*rkqc)(y,dydx,nvar,&x,h,eps,yscal,&hdid,&hnext,derivs);
		if (hdid == h) ++(*nok); else ++(*nbad);
		if ((x-x2)*(x2-x1) >= 0.0) {
			for (i=1;i<=nvar;i++) ystart[i]=y[i];
			if (kmax) {
				xp[++kount]=x;
				for (i=1;i<=nvar;i++) yp[i][kount]=y[i];
			}
			free_vector(dydx,1,nvar);
			free_vector(y,1,nvar);
			free_vector(yscal,1,nvar);
			return;
		}
		if (fabs(hnext) <= hmin) nrerror("Step size too small in ODEINT");
		h=hnext;
	}
	nrerror("Too many steps in routine ODEINT");
}

#undef MAXSTP
#undef TINY
