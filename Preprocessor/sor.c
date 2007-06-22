#include <math.h>

#define MAXITS 1000
#define EPS 1.e-5

void sor(a,b,c,d,e,f,u,jmax,rjac)
double **a,**b,**c,**d,**e,**f,**u,rjac;
int jmax;
{
	int n,l,j;
	double resid,omega,anormf,anorm;
	void nrerror();

	anormf=0.0;
	for (j=2;j<jmax;j++)
		for (l=2;l<jmax;l++)
			anormf += fabs(f[j][l]);
	omega=1.0;
	for (n=1;n<=MAXITS;n++) {
		anorm=0.0;
		for (j=2;j<jmax;j++)
			for (l=2;l<jmax;l++)
				if ((j+l)%2 == n%2) {
					resid=a[j][l]*u[j+1][l]
						+b[j][l]*u[j-1][l]
						+c[j][l]*u[j][l+1];
					resid += d[j][l]*u[j][l-1]
						+e[j][l]*u[j][l]-f[j][l];
					anorm += fabs(resid);
					u[j][l] -= omega*resid/e[j][l];
				}
		omega=(n == 1 ? 1.0/(1.0-0.5*rjac*rjac) :
			1.0/(1.0-0.25*rjac*rjac*omega));
		if (n > 1 && anorm < EPS*anormf) return;
	}
	nrerror("Maximum number of iterations exceeded in SOR");
}

#undef MAXITS
#undef EPS
