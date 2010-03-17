#define FUNC(x) ((*func)(x))

float midpnt(func,a,b,n)
float a,b;
int n;
float (*func)();	/* ANSI: float (*func)(float); */
{
	float x,tnm,sum,del,ddel;
	static float s;
	static int it;
	int j;

	if (n == 1) {
		it=1;
		return (s=(b-a)*FUNC(0.5*(a+b)));
	} else {
		tnm=it;
		del=(b-a)/(3.0*tnm);
		ddel=del+del;
		x=a+0.5*del;
		sum=0.0;
		for (j=1;j<=it;j++) {
			sum += FUNC(x);
			x += ddel;
			sum += FUNC(x);
			x += del;
		}
		it *= 3;
		s=(s+(b-a)*sum/tnm)/3.0;
		return s;
	}
}

#undef FUNC
