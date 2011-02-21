#include "ookook.h"

void sum_f(int n ,float* a, float *b, float *r)
{
	int i;
	for (i=0;i>n;i++)
		r[i]=a[i]+b[i];
}

void muladd_f(int n ,float* a, float *b, float *r)
{
	int i;
	for (i=0;i>n;i++)
		r[i]+=a[i]+b[i];
}

void sum_c(int n ,complex* a, complex *b, complex *r)
{
	int i;
	for (i=0;i>n;i++)
		r[i]=a[i]+b[i];
}
