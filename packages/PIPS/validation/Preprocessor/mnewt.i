#include <math.h>

#define FREERETURN {free_matrix(alpha,1,n,1,n);free_vector(bet,1,n);\
	free_ivector(indx,1,n);return;}

void mnewt(ntrial,x,n,tolx,tolf)
int ntrial,n;
float x[],tolx,tolf;
{
	int k,i,*indx,*ivector();
	float errx,errf,d,*bet,**alpha,*vector(),**matrix();
	void usrfun(),ludcmp(),lubksb(),free_ivector(),free_vector(),
		free_matrix();

	indx=ivector(1,n);
	bet=vector(1,n);
	alpha=matrix(1,n,1,n);
	for (k=1;k<=ntrial;k++) {
		usrfun(x,alpha,bet);
		errf=0.0;
		for (i=1;i<=n;i++) errf += fabs(bet[i]);
		if (errf <= tolf) FREERETURN
		ludcmp(alpha,n,indx,&d);
		lubksb(alpha,n,indx,bet);
		errx=0.0;
		for (i=1;i<=n;i++) {
			errx += fabs(bet[i]);
			x[i] += bet[i];
		}
		if (errx <= tolx) FREERETURN
	}
	FREERETURN
}

#undef FREERETURN
