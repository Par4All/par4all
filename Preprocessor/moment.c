#include <math.h>

void moment(data,n,ave,adev,sdev,svar,skew,curt)
int n;
float data[],*ave,*adev,*sdev,*svar,*skew,*curt;
{
	int j;
	float s,p;
	void nrerror();

	if (n <= 1) nrerror("n must be at least 2 in MOMENT");
	s=0.0;
	for (j=1;j<=n;j++) s += data[j];
	*ave=s/n;
	*adev=(*svar)=(*skew)=(*curt)=0.0;
	for (j=1;j<=n;j++) {
		*adev += fabs(s=data[j]-(*ave));
		*svar += (p=s*s);
		*skew += (p *= s);
		*curt += (p *= s);
	}
	*adev /= n;
	*svar /= (n-1);
	*sdev=sqrt(*svar);
	if (*svar) {
		*skew /= (n*(*svar)*(*sdev));
		*curt=(*curt)/(n*(*svar)*(*svar))-3.0;
	} else nrerror("No skew/kurtosis when variance = 0 (in MOMENT)");
}
