#include <stdio.h>
#include <stdlib.h>
#include "tools.h"

void sum_f(unsigned int n, float *res, float *a, float *b)
{
	unsigned int i;
	for (i=0;i<n;i++)
		res[i]=a[i]+b[i];
}

int main(int argc, char** argv)
{
	unsigned int n,i;
	float *a,*b,*res;
	if (argc < 2)
		return 1;
	n = atoi(argv[1]);
	a = (float*)malloc(n*sizeof(float));
	b = (float*)malloc(n*sizeof(float));
	res = (float*)malloc(n*sizeof(float));

	for (i=0; i < n; i++)
	{
		a[i] = i;
		b[i] = n-i;
	}

	sum_f(n,res,a,b);
	//print_array_float("a",a,n);
	//print_array_float("b",b,n);
	print_array_float("res",res,n);

	return 0;
}
