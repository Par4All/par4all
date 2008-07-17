#include <math.h>

#define M 714025
#define IA 1366
#define IC 150889

float ran2(idum)
long *idum;
{
	static long iy,ir[98];
	static int iff=0;
	int j;
	void nrerror();

	if (*idum < 0 || iff == 0) {
		iff=1;
		if ((*idum=(IC-(*idum)) % M) < 0) *idum = -(*idum);
		for (j=1;j<=97;j++) {
			*idum=(IA*(*idum)+IC) % M;
			ir[j]=(*idum);
		}
		*idum=(IA*(*idum)+IC) % M;
		iy=(*idum);
	}
	j=1 + 97.0*iy/M;
	if (j > 97 || j < 1) nrerror("RAN2: This cannot happen.");
	iy=ir[j];
	*idum=(IA*(*idum)+IC) % M;
	ir[j]=(*idum);
	return (float) iy/M;
}

#undef M
#undef IA
#undef IC
