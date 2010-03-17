#include <math.h>

#define PI 3.141592653589793

void chebft(a,b,c,n,func)
float a,b,c[];
float (*func)();	/* ANSI: float (*func)(float); */
int n;
{
	int k,j;
	float fac,bpa,bma,*f,*vector();
	void free_vector();

	f=vector(0,n-1);
	bma=0.5*(b-a);
	bpa=0.5*(b+a);
	for (k=0;k<n;k++) {
		float y=cos(PI*(k+0.5)/n);
		f[k]=(*func)(y*bma+bpa);
	}
	fac=2.0/n;
	for (j=0;j<n;j++) {
		double sum=0.0;
		for (k=0;k<n;k++)
			sum += f[k]*cos(PI*j*(k+0.5)/n);
		c[j]=fac*sum;
	}
	free_vector(f,0,n-1);
}

#undef PI
