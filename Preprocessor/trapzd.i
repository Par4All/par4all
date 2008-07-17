#define FUNC(x) ((*func)(x))

float trapzd(func,a,b,n)
float a,b;
float (*func)();	/* ANSI: float (*func)(float); */
int n;
{
	float x,tnm,sum,del;
	static float s;
	static int it;
	int j;

	if (n == 1) {
		it=1;
		return (s=0.5*(b-a)*(FUNC(a)+FUNC(b)));
	} else {
		tnm=it;
		del=(b-a)/tnm;
		x=a+0.5*del;
		for (sum=0.0,j=1;j<=it;j++,x+=del) sum += FUNC(x);
		it *= 2;
		s=0.5*(s+(b-a)*sum/tnm);
		return s;
	}
}
