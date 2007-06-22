#include <math.h>

float evlmem(fdt,cof,m,pm)
float fdt,cof[],pm;
int m;
{
	int i;
	float sumr=1.0,sumi=0.0;
	double wr=1.0,wi=0.0,wpr,wpi,wtemp,theta;

	theta=6.28318530717959*fdt;
	wpr=cos(theta);
	wpi=sin(theta);
	for (i=1;i<=m;i++) {
		wr=(wtemp=wr)*wpr-wi*wpi;
		wi=wi*wpr+wtemp*wpi;
		sumr -= cof[i]*wr;
		sumi -= cof[i]*wi;
	}
	return pm/(sumr*sumr+sumi*sumi);
}
