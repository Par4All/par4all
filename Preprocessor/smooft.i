#include <math.h>

void smooft(y,n,pts)
float y[],pts;
int n;
{
	int nmin,m=2,mo2,k,j;
	float yn,y1,rn1,fac,cnst;
	void realft();

	nmin=n+(int) (2.0*pts+0.5);
	while (m < nmin) m *= 2;
	cnst=pts/m,cnst=cnst*cnst;
	y1=y[1];
	yn=y[n];
	rn1=1.0/(n-1);
	for (j=1;j<=n;j++)
		y[j] += (-rn1*(y1*(n-j)+yn*(j-1)));
	for (j=n+1;j<=m;j++) y[j]=0.0;
	mo2=m >> 1;
	realft(y,mo2,1);
	y[1] /= mo2;
	fac=1.0;
	for (j=1;j<mo2;j++) {
		k=2*j+1;
		if (fac) {
			if ( (fac=(1.0-cnst*j*j)/mo2) < 0.0) fac=0.0;
			y[k]=fac*y[k];
			y[k+1]=fac*y[k+1];
		} else  y[k+1]=y[k]=0.0;
	}
	if ( (fac=(1.0-0.25*pts*pts)/mo2) < 0.0) fac=0.0;
	y[2] *= fac;
	realft(y,mo2,-1);
	for (j=1;j<=n;j++)
		y[j] += rn1*(y1*(n-j)+yn*(j-1));
}
