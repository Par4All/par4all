#include <math.h>

#define EPS 1.0e-5
#define JMAX 20

float qtrap(func,a,b)
float a,b;
float (*func)();
{
	int j;
	float s,olds,trapzd();
	void nrerror();

	olds = -1.0e30;
	for (j=1;j<=JMAX;j++) {
		s=trapzd(func,a,b,j);
		if (fabs(s-olds) < EPS*fabs(olds)) return s;
		olds=s;
	}
	nrerror("Too many steps in routine QTRAP");
}

#undef EPS
#undef JMAX
