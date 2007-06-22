#define FUNC(x) ((*funk)(1.0/(x))/((x)*(x)))

float midinf(funk,aa,bb,n)
float aa,bb;
int n;
float (*funk)();	/* ANSI: float (*funk)(float); */
{
	float x,tnm,sum,del,ddel,b,a;
	static float s;
	static int it;
	int j;

	b=1.0/aa;
	a=1.0/bb;
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
		return (s=(s+(b-a)*sum/tnm)/3.0);
	}
}

#undef FUNC
