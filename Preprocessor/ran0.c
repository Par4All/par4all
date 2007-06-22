#include <stdlib.h>
#include <math.h>

float ran0(idum)
int *idum;
{
	static float y,maxran,v[98];
	float dum;
	static int iff=0;
	int j;
	unsigned i,k;
	void nrerror();

	if (*idum < 0 || iff == 0) {
		iff=1;
		i=2;
		do {
			k=i;
			i<<=1;
		} while (i);
		maxran=k;
		srand(*idum);
		*idum=1;
		for (j=1;j<=97;j++) dum=rand();
		for (j=1;j<=97;j++) v[j]=rand();
		y=rand();
	}
	j=1+97.0*y/maxran;
	if (j > 97 || j < 1) nrerror("RAN0: This cannot happen.");
	y=v[j];
	v[j]=rand();
	return y/maxran;
}
