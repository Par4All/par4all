#include <math.h>

#define ITMAX 20
#define TINY 1.0e-6

void qroot(p,n,b,c,eps)
float p[],*b,*c,eps;
int n;
{
	int iter,i;
	float sc,sb,s,rc,rb,r,dv,delc,delb;
	float *q,*qq,*rem,*vector();
	float d[3];
	void poldiv(),nrerror(),free_vector();

	q=vector(0,n);
	qq=vector(0,n);
	rem=vector(0,n);
	d[2]=1.0;
	for (iter=1;iter<=ITMAX;iter++) {
		d[1]=(*b);
		d[0]=(*c);
		poldiv(p,n,d,2,q,rem);
		s=rem[0];
		r=rem[1];
		poldiv(q,(n-1),d,2,qq,rem);
		sc = -rem[0];
		rc = -rem[1];
		for (i=n-1;i>=0;i--) q[i+1]=q[i];
		q[0]=0.0;
		poldiv(q,n,d,2,qq,rem);
		sb = -rem[0];
		rb = -rem[1];
		dv=1.0/(sb*rc-sc*rb);
		*b += (delb=(r*sc-s*rc)*dv);
		*c += (delc=(-r*sb+s*rb)*dv);
		if ((fabs(delb) <= eps*fabs(*b) || fabs(*b) < TINY)
		&&  (fabs(delc) <= eps*fabs(*c) || fabs(*c) < TINY)) {
			free_vector(rem,0,n);
			free_vector(qq,0,n);
			free_vector(q,0,n);
			return;
		}
	}
	nrerror("Too many iterations in routine QROOT");
}

#undef ITMAX
#undef TINY
