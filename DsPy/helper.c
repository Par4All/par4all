#include "helper.h"
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>

void finit(int n, float *a) {
	int i;
	for(i=0;i<n;i++) *a++=(float)1000.f*drand48();
}
void cshow(int n, float complex *a) {
	printf("%f %f\n",crealf(a[n/2]),cimagf(a[n/2]));
}
void fshow(int n, float *a) {
	printf("%f\n",a[n/2]);
}

