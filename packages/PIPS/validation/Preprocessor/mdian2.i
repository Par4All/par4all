#include <math.h>

#define BIG 1.0e30
#define AFAC 1.5
#define AMP 1.5

void mdian2(x,n,xmed)
int n;
float x[],*xmed;
{
	int np,nm,j;
	float xx,xp,xm,sumx,sum,eps,stemp,dum,ap,am,aa,a;

	a=0.5*(x[1]+x[n]);
	eps=fabs(x[n]-x[1]);
	am = -(ap=BIG);
	for (;;) {
		sum=sumx=0.0;
		np=nm=0;
		xm = -(xp=BIG);
		for (j=1;j<=n;j++) {
			xx=x[j];
			if (xx != a) {
				if (xx > a) {
					++np;
					if (xx < xp) xp=xx;
				} else if (xx < a) {
					++nm;
					if (xx > xm) xm=xx;
				}
				sum += dum=1.0/(eps+fabs(xx-a));
				sumx += xx*dum;
			}
		}
		stemp=(sumx/sum)-a;
		if (np-nm >= 2) {
			am=a;
			aa =  stemp < 0.0 ? xp : xp+stemp*AMP;
			if (aa > ap) aa=0.5*(a+ap);
			eps=AFAC*fabs(aa-a);
			a=aa;
		} else if (nm-np >= 2) {
			ap=a;
			aa = stemp > 0.0 ? xm : xm+stemp*AMP;
			if (aa < am) aa=0.5*(a+am);
			eps=AFAC*fabs(aa-a);
			a=aa;
		} else {
			if (n % 2 == 0) {
				*xmed = 0.5*(np == nm ? xp+xm : np > nm ? a+xp : xm+a);
			} else {
				*xmed = np == nm ? a : np > nm ? xp : xm;
			}
			return;
		}
	}
}

#undef BIG
#undef AFAC
#undef AMP
