#include <math.h>

#define ACC 40.0
#define BIGNO 1.0e10
#define BIGNI 1.0e-10

float bessj(n,x)
int n;
float x;
{
	int j,jsum,m;
	float ax,bj,bjm,bjp,sum,tox,ans;
	float bessj0(),bessj1();
	void nrerror();

	if (n < 2) nrerror("Index n less than 2 in BESSJ");
	ax=fabs(x);
	if (ax == 0.0)
		return 0.0;
	else if (ax > (float) n) {
		tox=2.0/ax;
		bjm=bessj0(ax);
		bj=bessj1(ax);
		for (j=1;j<n;j++) {
			bjp=j*tox*bj-bjm;
			bjm=bj;
			bj=bjp;
		}
		ans=bj;
	} else {
		tox=2.0/ax;
		m=2*((n+(int) sqrt(ACC*n))/2);
		jsum=0;
		bjp=ans=sum=0.0;
		bj=1.0;
		for (j=m;j>0;j--) {
			bjm=j*tox*bj-bjp;
			bjp=bj;
			bj=bjm;
			if (fabs(bj) > BIGNO) {
				bj *= BIGNI;
				bjp *= BIGNI;
				ans *= BIGNI;
				sum *= BIGNI;
			}
			if (jsum) sum += bj;
			jsum=!jsum;
			if (j == n) ans=bjp;
		}
		sum=2.0*sum-bj;
		ans /= sum;
	}
	return  x < 0.0 && n%2 == 1 ? -ans : ans;
}

#undef ACC
#undef BIGNO
#undef BIGNI
