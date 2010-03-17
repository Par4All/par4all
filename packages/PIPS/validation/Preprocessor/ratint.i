#include <math.h>

#define TINY 1.0e-25
#define FREERETURN {free_vector(d,1,n);free_vector(c,1,n);return;}

void ratint(xa,ya,n,x,y,dy)
float xa[],ya[],x,*y,*dy;
int n;
{
	int m,i,ns=1;
	float w,t,hh,h,dd,*c,*d,*vector();
	void nrerror(),free_vector();

	c=vector(1,n);
	d=vector(1,n);
	hh=fabs(x-xa[1]);
	for (i=1;i<=n;i++) {
		h=fabs(x-xa[i]);
		if (h == 0.0) {
			*y=ya[i];
			*dy=0.0;
			FREERETURN
		} else if (h < hh) {
			ns=i;
			hh=h;
		}
		c[i]=ya[i];
		d[i]=ya[i]+TINY;
	}
	*y=ya[ns--];
	for (m=1;m<n;m++) {
		for (i=1;i<=n-m;i++) {
			w=c[i+1]-d[i];
			h=xa[i+m]-x;
			t=(xa[i]-x)*d[i]/h;
			dd=t-c[i+1];
			if (dd == 0.0) nrerror("Error in routine RATINT");
			dd=w/dd;
			d[i]=c[i+1]*dd;
			c[i]=t*dd;
		}
		*y += (*dy=(2*ns < (n-m) ? c[ns+1] : d[ns--]));
	}
	FREERETURN
}

#undef TINY
#undef FREERETURN
