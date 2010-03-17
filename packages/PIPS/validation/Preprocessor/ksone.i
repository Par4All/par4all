#include <math.h>

static float maxarg1,maxarg2;
#define MAX(a,b) (maxarg1=(a),maxarg2=(b),(maxarg1) > (maxarg2) ?\
	(maxarg1) : (maxarg2))

void ksone(data,n,func,d,prob)
float data[],*d,*prob;
float (*func)();	/* ANSI: float (*func)(float); */
int n;
{
	int j;
	float fo=0.0,fn,ff,en,dt;
	void sort();
	float probks();

	sort(n,data);
	en=n;
	*d=0.0;
	for (j=1;j<=n;j++) {
		fn=j/en;
		ff=(*func)(data[j]);
		dt = MAX(fabs(fo-ff),fabs(fn-ff));
		if (dt > *d) *d=dt;
		fo=fn;
	}
	*prob=probks(sqrt(en)*(*d));
}

#undef MAX
