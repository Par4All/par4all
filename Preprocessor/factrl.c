#include <math.h>

float factrl(n)
int n;
{
	static int ntop=4;
	static float a[33]={1.0,1.0,2.0,6.0,24.0};
	int j;
	float gammln();
	void nrerror();

	if (n < 0) nrerror("Negative factorial in routine FACTRL");
	if (n > 32) return exp(gammln(n+1.0));
	while (ntop<n) {
		j=ntop++;
		a[ntop]=a[j]*ntop;
	}
	return a[n];
}
