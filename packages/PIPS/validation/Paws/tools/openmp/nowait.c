#include <math.h>

void a92(int n, float *a, float *b, float *c, float *y, float *z) {

	int i;
	
	for (i=0; i<n; i++)
		c[i] = (a[i] + b[i]) / 2.0;

	for (i=0; i<n; i++)
		z[i] = sqrt(c[i]);

	for (i=1; i<=n; i++)
		y[i] = z[i-1] + a[i];
	
}
