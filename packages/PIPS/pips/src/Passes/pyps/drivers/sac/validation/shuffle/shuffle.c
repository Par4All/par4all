#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "SIMD.h"
#include "SIMD_types.h"

void shuffle(int n, float r[n], float r3[n], float r2[n], float a[n], float b[n])
{
	int i;
	for (i=0; i<n; i += 4)
	{
		r[i] = a[i]+b[i];
		r[i+1] = a[i+1]+b[i+1];
		r[i+2] = a[i+2]+b[i+2];
		r[i+3] = a[i+3]+b[i+3];

		r2[i] = a[i]+b[i+2];
		r2[i+1] = a[i+1]+b[i+1];
		r2[i+2] = a[i+2]+b[i+3];
		r2[i+3] = a[i+3]+b[i];

		r3[i] = a[i]+b[i];
		r3[i+1] = a[i+1]+b[i+3];
		r3[i+2] = a[i+2]+b[i+2];
		r3[i+3] = a[i+3]+b[i+1];
	}
}

int main(int argc, char** argv)
{
	int n = (argc>1)?atoi(argv[0]):10;
	float* r = (float*)malloc(sizeof(float)*n);
	float* r2 = (float*)malloc(sizeof(float)*n);
	float* r3 = (float*)malloc(sizeof(float)*n);
	float* a = (float*)malloc(sizeof(float)*n);
	float* b = (float*)malloc(sizeof(float)*n);
	int i;

	for (i=0;i<n;i++)
	{
		a[i] = i;
		b[i] = i;
	}
	shuffle(n, r, r2, r3, a, b);

	for (i=0;i<n;i++)
		printf("%f %f\n", r[i], r2[i]);
	return 0;
}
