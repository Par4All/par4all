#include <math.h>

void cosft(y,n,isign)
float y[];
int n,isign;
{
	int j,m,n2;
	float enf0,even,odd,sum,sume,sumo,y1,y2;
	double theta,wi=0.0,wr=1.0,wpi,wpr,wtemp;
	void realft();

	theta=3.14159265358979/(double) n;
	wtemp=sin(0.5*theta);
	wpr = -2.0*wtemp*wtemp;
	wpi=sin(theta);
	sum=y[1];
	m=n >> 1;
	n2=n+2;
	for (j=2;j<=m;j++) {
		wr=(wtemp=wr)*wpr-wi*wpi+wr;
		wi=wi*wpr+wtemp*wpi+wi;
		y1=0.5*(y[j]+y[n2-j]);
		y2=(y[j]-y[n2-j]);
		y[j]=y1-wi*y2;
		y[n2-j]=y1+wi*y2;
		sum += wr*y2;
	}
	realft(y,m,1);
	y[2]=sum;
	for (j=4;j<=n;j+=2) {
		sum += y[j];
		y[j]=sum;
	}
	if (isign == -1) {
		even=y[1];
		odd=y[2];
		for (j=3;j<=n-1;j+=2) {
			even += y[j];
			odd += y[j+1];
		}
		enf0=2.0*(even-odd);
		sumo=y[1]-enf0;
		sume=(2.0*odd/n)-sumo;
		y[1]=0.5*enf0;
		y[2] -= sume;
		for (j=3;j<=n-1;j+=2) {
			y[j] -= sumo;
			y[j+1] -= sume;
		}
	}
}
