#include <math.h>

#define ITMAX 100
#define EPS 3.0e-7

float betacf(a,b,x)
float a,b,x;
{
	float qap,qam,qab,em,tem,d;
	float bz,bm=1.0,bp,bpp;
	float az=1.0,am=1.0,ap,app,aold;
	int m;
	void nrerror();

	qab=a+b;
	qap=a+1.0;
	qam=a-1.0;
	bz=1.0-qab*x/qap;
	for (m=1;m<=ITMAX;m++) {
		em=(float) m;
		tem=em+em;
		d=em*(b-em)*x/((qam+tem)*(a+tem));
		ap=az+d*am;
		bp=bz+d*bm;
		d = -(a+em)*(qab+em)*x/((qap+tem)*(a+tem));
		app=ap+d*az;
		bpp=bp+d*bz;
		aold=az;
		am=ap/bpp;
		bm=bp/bpp;
		az=app/bpp;
		bz=1.0;
		if (fabs(az-aold) < (EPS*fabs(az))) return az;
	}
	nrerror("a or b too big, or ITMAX too small in BETACF");
}

#undef ITMAX
#undef EPS
