#include <math.h>

#define ACC 40.0
#define BIGNO 1.0e10
#define BIGNI 1.0e-10

float bessi(n,x)
int n;
float x;
{
	int j;
	float bi,bim,bip,tox,ans;
	float bessi0();
	void nrerror();

	if (n < 2) nrerror("Index n less than 2 in BESSI");
	if (x == 0.0)
		return 0.0;
	else {
		tox=2.0/fabs(x);
		bip=ans=0.0;
		bi=1.0;
		for (j=2*(n+(int) sqrt(ACC*n));j>0;j--) {
			bim=bip+j*tox*bi;
			bip=bi;
			bi=bim;
			if (fabs(bi) > BIGNO) {
				ans *= BIGNI;
				bi *= BIGNI;
				bip *= BIGNI;
			}
			if (j == n) ans=bip;
		}
		ans *= bessi0(x)/bi;
		return  x < 0.0 && n%2 == 1 ? -ans : ans;
	}
}

#undef ACC
#undef BIGNO
#undef BIGNI
