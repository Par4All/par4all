#include <math.h>

#define EPS 1.0e-6
#define JMAX 14
#define JMAXP JMAX+1
#define K 5

float qromo(func,a,b,choose)
float a,b;
float (*func)();
float (*choose)();	/* ANSI: float choose(float(*)(float),float,float,int); */
{
	int j;
	float ss,dss,h[JMAXP+1],s[JMAXP+1];
	void polint(),nrerror();

	h[1]=1.0;
	for (j=1;j<=JMAX;j++) {
		s[j]=(*choose)(func,a,b,j);
		if (j >= K) {
			polint(&h[j-K],&s[j-K],K,0.0,&ss,&dss);
			if (fabs(dss) < EPS*fabs(ss)) return ss;
		}
		s[j+1]=s[j];
		h[j+1]=h[j]/9.0;
	}
	nrerror("Too many steps in routing QROMO");
}

#undef EPS
#undef JMAX
#undef JMAXP
#undef K
