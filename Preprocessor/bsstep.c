#include <math.h>

#define IMAX 11
#define NUSE 7
#define SHRINK 0.95
#define GROW 1.2

float **d=0,*x=0;	/* defining declaration */

void bsstep(y,dydx,nv,xx,htry,eps,yscal,hdid,hnext,derivs)
float y[],dydx[],*xx,htry,eps,yscal[],*hdid,*hnext;
void (*derivs)();
int nv;
{
	int i,j;
	float xsav,xest,h,errmax,temp;
	float *ysav,*dysav,*yseq,*yerr,*vector(),**matrix();
	static int nseq[IMAX+1]={0,2,4,6,8,12,16,24,32,48,64,96};
	void mmid(),rzextr(),nrerror(),free_matrix(),free_vector();

	ysav=vector(1,nv);
	dysav=vector(1,nv);
	yseq=vector(1,nv);
	yerr=vector(1,nv);
	x=vector(1,IMAX);
	d=matrix(1,nv,1,NUSE);
	h=htry;
	xsav=(*xx);
	for (i=1;i<=nv;i++) {
		ysav[i]=y[i];
		dysav[i]=dydx[i];
	}
	for (;;) {
		for (i=1;i<=IMAX;i++) {
			mmid(ysav,dysav,nv,xsav,h,nseq[i],yseq,derivs);
			xest=(temp=h/nseq[i],temp*temp);
			rzextr(i,xest,yseq,y,yerr,nv,NUSE);
			errmax=0.0;
			for (j=1;j<=nv;j++)
				if (errmax < fabs(yerr[j]/yscal[j]))
					errmax=fabs(yerr[j]/yscal[j]);
			errmax /= eps;
			if (errmax < 1.0) {
				*xx += h;
				*hdid=h;
				*hnext = i==NUSE? h*SHRINK : i==NUSE-1?
					h*GROW : (h*nseq[NUSE-1])/nseq[i];
				free_matrix(d,1,nv,1,NUSE);
				free_vector(x,1,IMAX);
				free_vector(yerr,1,nv);
				free_vector(yseq,1,nv);
				free_vector(dysav,1,nv);
				free_vector(ysav,1,nv);
				return;
			}
		}
		h *= 0.25;
		for (i=1;i<=(IMAX-NUSE)/2;i++) h /= 2.0;
		if ((*xx+h) == (*xx)) nrerror("Step size underflow in BSSTEP");
	}
}

#undef IMAX
#undef NUSE
#undef SHRINK
#undef GROW
