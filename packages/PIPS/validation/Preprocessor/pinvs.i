#include <math.h>

void pinvs(ie1,ie2,je1,jsf,jc1,k,c,s)
int ie1,ie2,je1,jsf,jc1,k;
float ***c,**s;
{
	int js1,jpiv,jp,je2,jcoff,j,irow,ipiv,id,icoff,i,*indxr,*ivector();
	float pivinv,piv,dum,big,*pscl,*vector();
	void nrerror(),free_vector(),free_ivector();

	indxr=ivector(ie1,ie2);
	pscl=vector(ie1,ie2);
	je2=je1+ie2-ie1;
	js1=je2+1;
	for (i=ie1;i<=ie2;i++) {
		big=0.0;
		for (j=je1;j<=je2;j++)
			if (fabs(s[i][j]) > big) big=fabs(s[i][j]);
		if (big == 0.0) nrerror("Singular matrix - row all 0, in PINVS");
		pscl[i]=1.0/big;
		indxr[i]=0;
	}
	for (id=ie1;id<=ie2;id++) {
		piv=0.0;
		for (i=ie1;i<=ie2;i++) {
			if (indxr[i] == 0) {
				big=0.0;
				for (j=je1;j<=je2;j++) {
					if (fabs(s[i][j]) > big) {
						jp=j;
						big=fabs(s[i][j]);
					}
				}
				if (big*pscl[i] > piv) {
					ipiv=i;
					jpiv=jp;
					piv=big*pscl[i];
				}
			}
		}
		if (s[ipiv][jpiv] == 0.0) nrerror("Singular matrix in routine PINVS");
		indxr[ipiv]=jpiv;
		pivinv=1.0/s[ipiv][jpiv];
		for (j=je1;j<=jsf;j++) s[ipiv][j] *= pivinv;
		s[ipiv][jpiv]=1.0;
		for (i=ie1;i<=ie2;i++) {
			if (indxr[i] != jpiv) {
				if (s[i][jpiv]) {
					dum=s[i][jpiv];
					for (j=je1;j<=jsf;j++)
						s[i][j] -= dum*s[ipiv][j];
					s[i][jpiv]=0.0;
				}
			}
		}
	}
	jcoff=jc1-js1;
	icoff=ie1-je1;
	for (i=ie1;i<=ie2;i++) {
		irow=indxr[i]+icoff;
		for (j=js1;j<=jsf;j++) c[irow][j+jcoff][k]=s[i][j];
	}
	free_vector(pscl,ie1,ie2);
	free_ivector(indxr,ie1,ie2);
}
