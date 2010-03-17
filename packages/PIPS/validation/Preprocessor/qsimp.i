#include <math.h>

#define EPS 1.0e-6
#define JMAX 20

float qsimp(func,a,b)
float a,b;
float (*func)();
{
	int j;
	float s,st,ost,os,trapzd();
	void nrerror();

	ost = os =  -1.0e30;
	for (j=1;j<=JMAX;j++) {
		st=trapzd(func,a,b,j);
		s=(4.0*st-ost)/3.0;
		if (fabs(s-os) < EPS*fabs(os)) return s;
		os=s;
		ost=st;
	}
	nrerror("Too many steps in routine QSIMP");
}

#undef EPS
#undef JMAX
