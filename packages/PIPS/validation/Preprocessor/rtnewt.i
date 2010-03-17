#include <math.h>

#define JMAX 20

float rtnewt(funcd,x1,x2,xacc)
float x1,x2,xacc;
void (*funcd)();	/* ANSI: void (*funcd)(float,float *,float *); */
{
	int j;
	float df,dx,f,rtn;
	void nrerror();

	rtn=0.5*(x1+x2);
	for (j=1;j<=JMAX;j++) {
		(*funcd)(rtn,&f,&df);
		dx=f/df;
		rtn -= dx;
		if ((x1-rtn)*(rtn-x2) < 0.0)
			nrerror("Jumped out of brackets in RTNEWT");
		if (fabs(dx) < xacc) return rtn;
	}
	nrerror("Maximum number of iterations exceeded in RTNEWT");
}

#undef JMAX
