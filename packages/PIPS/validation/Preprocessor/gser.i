#include <math.h>

#define ITMAX 100
#define EPS 3.0e-7

void gser(gamser,a,x,gln)
float a,x,*gamser,*gln;
{
	int n;
	float sum,del,ap;
	float gammln();
	void nrerror();

	*gln=gammln(a);
	if (x <= 0.0) {
		if (x < 0.0) nrerror("x less than 0 in routine GSER");
		*gamser=0.0;
		return;
	} else {
		ap=a;
		del=sum=1.0/a;
		for (n=1;n<=ITMAX;n++) {
			ap += 1.0;
			del *= x/ap;
			sum += del;
			if (fabs(del) < fabs(sum)*EPS) {
				*gamser=sum*exp(-x+a*log(x)-(*gln));
				return;
			}
		}
		nrerror("a too large, ITMAX too small in routine GSER");
		return;
	}
}

#undef ITMAX
#undef EPS
