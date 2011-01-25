#include <stdio.h>
#include <stdlib.h>

void sum_f(float* a, float *b, float *r, int n)
{
	int i;
	for (i=0;i<n;i++)
		r[i]=a[i]+b[i];
}

void mul_f(float* a, float *b, float *r, int n)
{
	int i;
	for (i=0;i<n;i++)
		r[i]=a[i]*b[i];
}

void subs_f(float* a, float *b, float *r, int n)
{
	int i;
	for (i=0;i<n;i++)
		r[i]=a[i]-b[i];
}

void div_f(float* a, float *b, float *r, int n)
{
	int i;
	for (i=0;i<n;i++)
		r[i]=a[i]/b[i];
}

void muladd_f(float* a, float* b, float* r, int n)
{
	int i;
	for (i=0;i<n;i++)
		r[i]+=a[i]*b[i];
}

void umin_f(float* a, float* b, float* r, int n)
{
	int i;
	for (i=0;i<n;i++)
		r[i] = -(b[i]);
}

typedef void (*paction)(float*,float*,float*,int);

paction _funcs[] = {sum_f,mul_f,subs_f,div_f,muladd_f,umin_f};

int main(int argc, char** argv)
{
	float *a,*b,*r;
	int n,fi;
	paction f;
	if (argc < 4)
	{
		fprintf(stderr, "Usage: %s operation table_size data_file\n", argv[0]);
		fprintf(stderr, "where operation is :\n");
		fprintf(stderr, "0: sum\n1: mul\n2: sub\n3: div\n4: muladd\n5: umin\n");
		return 1;
	}
	n = atoi(argv[2]);
	a = malloc(n*sizeof(float));
	b = malloc(n*sizeof(float));
	r = malloc(n*sizeof(float));

	if (a == 0 || b == 0 || r == 0)
	{
		fprintf(stderr, "Unable to allocate memory !\n");
		return 2;
	}
	init_data_file(argv[3]);
	init_data_float(a, n);
	init_data_float(b, n);
	init_data_float(r, n);
	close_data_file();

	print_array_float("a", a, n);
	print_array_float("b", b, n);
	print_array_float("result_before", r, n);

	fi = atoi(argv[1]);
	if (fi >= sizeof(_funcs)/sizeof(paction))
	{
		fprintf(stderr, "Invalid operation number\n");
		return 4;
	}
	/*f = _funcs[fi];
	f(a,b,r,n);*/
	switch (fi)
	{
		case 0: sum_f(a,b,r,n); break;
		case 1: mul_f(a,b,r,n); break;
		case 2: subs_f(a,b,r,n); break;
		case 3: div_f(a,b,r,n); break;
		case 4: muladd_f(a,b,r,n); break;
		case 5: umin_f(a,b,r,n); break;
		default: break;
	}

	print_array_float("result", r, n);
	return 0;
}
