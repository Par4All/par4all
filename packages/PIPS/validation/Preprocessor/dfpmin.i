#include <math.h>

#define ITMAX 200
#define EPS 1.0e-10

void dfpmin(p,n,ftol,iter,fret,func,dfunc)
float p[],ftol,*fret,(*func)();
void (*dfunc)();
int n,*iter;
{
	int j,i,its;
	float fp,fae,fad,fac;
	float *xi,*g,*dg,*hdg,*vector();
	float **hessin,**matrix();
	void linmin(),nrerror(),free_matrix(),free_vector();

	hessin=matrix(1,n,1,n);
	xi=vector(1,n);
	g=vector(1,n);
	dg=vector(1,n);
	hdg=vector(1,n);
	fp=(*func)(p);
	(*dfunc)(p,g);
	for (i=1;i<=n;i++) {
		for (j=1;j<=n;j++) hessin[i][j]=0.0;
		hessin[i][i]=1.0;
		xi[i] = -g[i];
	}
	for (its=1;its<=ITMAX;its++) {
		*iter=its;
		linmin(p,xi,n,fret,func);
		if (2.0*fabs(*fret-fp) <= ftol*(fabs(*fret)+fabs(fp)+EPS)) {
			free_vector(hdg,1,n);
			free_vector(dg,1,n);
			free_vector(g,1,n);
			free_vector(xi,1,n);
			free_matrix(hessin,1,n,1,n);
			return;
		}
		fp=(*fret);
		for (i=1;i<=n;i++) dg[i]=g[i];
		*fret=(*func)(p);
		(*dfunc)(p,g);
		for (i=1;i<=n;i++) dg[i]=g[i]-dg[i];
		for (i=1;i<=n;i++) {
			hdg[i]=0.0;
			for (j=1;j<=n;j++) hdg[i] += hessin[i][j]*dg[j];
		}
		fac=fae=0.0;
		for (i=1;i<=n;i++) {
			fac += dg[i]*xi[i];
			fae += dg[i]*hdg[i];
		}
		fac=1.0/fac;
		fad=1.0/fae;
		for (i=1;i<=n;i++) dg[i]=fac*xi[i]-fad*hdg[i];
		for (i=1;i<=n;i++)
			for (j=1;j<=n;j++)
				hessin[i][j] += fac*xi[i]*xi[j]
					-fad*hdg[i]*hdg[j]+fae*dg[i]*dg[j];
		for (i=1;i<=n;i++) {
			xi[i]=0.0;
			for (j=1;j<=n;j++) xi[i] -= hessin[i][j]*g[j];
		}
	}
	nrerror("Too many iterations in DFPMIN");
}

#undef ITMAX
#undef EPS
