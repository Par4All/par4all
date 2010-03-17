#include <math.h>

#define JJ 50
#define KK 6
#define NRR 32  /* NRR=2 to the power (KK-1) */
#define MAXITS 100

void adi(a,b,c,d,e,f,g,u,jmax,k,alpha,beta,eps)
double **a,**b,**c,**d,**e,**f,**g,**u,alpha,beta,eps;
int jmax,k;
{
	int i,nr,nits,next,n,l,kits,k1,j,twopwr;
	double **psi,rfact,resid,disc,anormg,anorm,ab;
	double *aa,*bb,*cc,*rr,*uu,**s,*r,*alph,*bet;
	double **dmatrix(),*dvector();
	void free_dmatrix(),free_dvector(),nrerror(),tridag();

	if (jmax > JJ) nrerror("in ADI, increase JJ");
	if (k > KK-1)  nrerror("in ADI, increase KK");
	psi=dmatrix(1,JJ,1,JJ);
	s=dmatrix(1,NRR,1,KK);
	aa=dvector(1,JJ);
	bb=dvector(1,JJ);
	cc=dvector(1,JJ);
	rr=dvector(1,JJ);
	uu=dvector(1,JJ);
	r=dvector(1,NRR);
	alph=dvector(1,KK);
	bet=dvector(1,KK);
	k1=k+1;
	nr=1;
	for (i=1;i<=k;i++) nr *= 2;
	alph[1]=alpha;
	bet[1]=beta;
	for (j=1;j<=k;j++) {
		alph[j+1]=sqrt(alph[j]*bet[j]);
		bet[j+1]=0.5*(alph[j]+bet[j]);
	}
	s[1][1]=sqrt(alph[k1]*bet[k1]);
	for (j=1;j<=k;j++) {
		ab=alph[k1-j]*bet[k1-j];
		twopwr=1;
		for (i=1;i<=(j-1);i++) twopwr *= 2;
		for (n=1;n<=twopwr;n++) {
			disc=sqrt(s[n][j]*s[n][j]-ab);
			s[2*n][j+1]=s[n][j]+disc;
			s[2*n-1][j+1]=ab/s[2*n][j+1];
		}
	}
	for (n=1;n<=nr;n++) r[n]=s[n][k1];
	anormg=0.0;
	for (j=2;j<=jmax-1;j++) {
		for (l=2;l<=jmax-1;l++) {
			anormg += fabs(g[j][l]);
			psi[j][l] = -d[j][l]*u[j][l-1]
				+(r[1]-e[j][l])*u[j][l]-f[j][l]*u[j][l+1];
		}
	}
	nits=MAXITS/nr;
	for (kits=1;kits<=nits;kits++) {
		for (n=1;n<=nr;n++) {
			next = n == nr ? 1 : n+1;
			rfact=r[n]+r[next];
			for (l=2;l<=jmax-1;l++) {
				for (j=2;j<=jmax-1;j++) {
					aa[j-1]=a[j][l];
					bb[j-1]=b[j][l]+r[n];
					cc[j-1]=c[j][l];
					rr[j-1]=psi[j][l]-g[j][l];
				}
				tridag(aa,bb,cc,rr,uu,jmax-2);
				for (j=2;j<=jmax-1;j++)
					psi[j][l] = -psi[j][l]
						+2.0*r[n]*uu[j-1];
			}
			for (j=2;j<=jmax-1;j++) {
				for (l=2;l<=jmax-1;l++) {
					aa[l-1]=d[j][l];
					bb[l-1]=e[j][l]+r[n];
					cc[l-1]=f[j][l];
					rr[l-1]=psi[j][l];
				}
				tridag(aa,bb,cc,rr,uu,jmax-2);
				for (l=2;l<=jmax-1;l++) {
					u[j][l]=uu[l-1];
					psi[j][l] = -psi[j][l]+rfact*uu[l-1];
				}
			}
		}
		anorm=0.0;
		for (j=2;j<=jmax-1;j++)
			for (l=2;l<=jmax-1;l++) {
				resid=a[j][l]*u[j-1][l]
					+(b[j][l]+e[j][l])*u[j][l];
				resid += c[j][l]*u[j+1][l]+d[j][l]*u[j][l-1]
					+f[j][l]*u[j][l+1]+g[j][l];
				anorm += fabs(resid);
			}
		if (anorm < (eps*anormg)) {
			free_dvector(bet,1,KK);
			free_dvector(alph,1,KK);
			free_dvector(r,1,NRR);
			free_dvector(uu,1,JJ);
			free_dvector(rr,1,JJ);
			free_dvector(cc,1,JJ);
			free_dvector(bb,1,JJ);
			free_dvector(aa,1,JJ);
			free_dmatrix(s,1,NRR,1,KK);
			free_dmatrix(psi,1,JJ,1,JJ);
			return;
		}
	}
	nrerror("in ADI, too many iterations");
}

/* Double precision version of TRIDAG */
void tridag(a,b,c,r,u,n)
double *a,*b,*c,*r,*u;
int n;
{
	int j;
	double bet,*gam,*dvector();
	void nrerror(),free_dvector();

	gam=dvector(1,n);
	if (b[1] == 0.0) nrerror("error 1 in TRIDAG");
	bet=b[1];
	u[1]=r[1]/bet;
	for (j=2;j<=n;j++) {
		gam[j]=c[j-1]/bet;
		bet=b[j]-a[j]*gam[j];
		if (bet == 0.0) nrerror("error 2 in TRIDAG");
		u[j]=(r[j]-a[j]*u[j-1])/bet;
	}
	for (j=n-1;j>=1;j--)
		u[j] -= gam[j+1]*u[j+1];
	free_dvector(gam,1,n);
}

#undef JJ
#undef KK
#undef NRR
#undef MAXITS






