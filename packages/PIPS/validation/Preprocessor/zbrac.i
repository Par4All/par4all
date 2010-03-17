#include <math.h>

#define FACTOR 1.6
#define NTRY 50

int zbrac(func,x1,x2)
float *x1,*x2;
float (*func)();	/* ANSI: float (*func)(float); */
{
	int j;
	float f1,f2;
	void nrerror();

	if (*x1 == *x2) nrerror("Bad initial range in ZBRAC");
	f1=(*func)(*x1);
	f2=(*func)(*x2);
	for (j=1;j<=NTRY;j++) {
		if (f1*f2 < 0.0) return 1;
		if (fabs(f1) < fabs(f2))
			f1=(*func)(*x1 += FACTOR*(*x1-*x2));
		else
			f2=(*func)(*x2 += FACTOR*(*x2-*x1));
	}
	return 0;
}

#undef FACTOR
#undef NTRY
